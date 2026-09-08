from __future__ import annotations

"""Stage 7 — LLM Recourse Action Generation.

Reads the Stage 2 ``phase_eval_candidate`` serving panel (state-only) and the
Stage 6 ``policy_actions.parquet`` (for the RL reference policy on
reference conditions C6/C6X/C7/C8), runs the configured LLM backend across
conditions × modes, and emits a Stage 6-compatible action table that
Stage 8 will score with the Stage 6 simulator + Oracle substrate.

The stage is deliberately narrow:

* It does **not** simulate, score, or evaluate.  Stage 8 does that.
* It does **not** redefine the Oracle, the simulator, the candidate library,
  the temporal split, or the RL winner (per the Stage 7-9 LLM contract §1).
* It records every prompt, every raw LLM response, every parsed action, and
  every failure category so a reviewer can audit any specific LLM decision
  end to end.

Outputs (under ``data/final_freeze/stage7_llm_action_generation/``):

* ``llm_stage7_action_table.parquet`` — Stage 6 policy_actions-compatible
  schema (row_id, policy, candidate_id, action__* columns, plus Stage 7
  diagnostics).
* ``llm_stage7_prompt_manifest.json`` — frozen prompts, backend identity,
  reproducibility metadata.
* ``llm_stage7_failure_audit.csv`` — per-row failure taxonomy classification.
* ``llm_stage7_response_log.parquet`` — raw response text per request for
  full traceability.
* ``metadata.json`` — stage status, hashes, backend identity, condition
  matrix used.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_recourse.contracts.stage_paths import stage_dir, final_root
from credit_recourse.rl.common.actions import (
    active_config_hashes,
    assert_hashes_match,
    load_action_space,
    resolve_candidate_library_path,
)
from credit_recourse.rl.common.io import read_parquet_required, write_json

from .llm_backends import LLMBackend, LLMRequest, LLMResponse, LiveLLMBackend, make_backend
from .budget_contract import (
    BUDGET_AUDIT_COLUMNS,
    ACTION_BUDGET_CONTRACT_SCHEMA_VERSION,
    action_l1,
    budget_applies,
    disabled_budget_contract,
    make_action_budget_contract,
)
from .prompt_builder import (
    ICC_PROBE_SCHEMA_VERSION,
    ICC_PROBE_TARGET_FEATURE,
    build_icc_probe_prompt,
    build_prompt,
)
from .response_parser import (
    GROUNDING_MATCHER_VERSION,
    ParsedAction,
    parse_response,
    project_free_form_batch,
    to_failure_audit_frame,
    to_policy_actions_frame,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


STAGE7_CHECKPOINT_SCHEMA_VERSION = "stage7_llm_request_checkpoint_v1"
STAGE7_PROMPT_PAYLOAD_ARCHIVE_SCHEMA_VERSION = "stage7_prompt_payload_archive_v1"
DEFAULT_STAGE7_CHECKPOINT_NAME = "llm_stage7_response_checkpoint.jsonl"

# 2026-07-04 contract addendum (LLM789-008/009): IC-c exposes the firm NAME
# (research decision superseding-in-part contract v4 §6.1 identifier-only
# list) and the optional IC-c prior-knowledge probe emits the contract-v4
# icc_probe_* columns.  Runs carrying this key are verified against the
# addendum; legacy runs without it receive warnings only.
STAGE7_CONTRACT_ADDENDUM = "20260704_icc_firm_name_probe_v1"
ICC_PROBE_CHECKPOINT_NAME = "llm_stage7_icc_probe_checkpoint.jsonl"
ICC_PROBE_COLUMNS = [
    "icc_probe_response_raw", "icc_probe_value", "icc_contamination_flag",
    "icc_probe_parse_error", "icc_probe_rel_err", "icc_probe_panel_value",
]


def _json_dumps_stable(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _read_row_id_file(path: Path) -> list[int]:
    """Read a pre-registered Stage 7 row-id subset file.

    Accepted formats are CSV/TXT (with a ``row_id`` column or a single column)
    and Parquet (with a ``row_id`` column).  The function hard-fails on missing,
    null, non-integer, or duplicated row ids so an N5 pilot cannot silently drift
    from its pre-registered sample.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Stage 7 row-id subset file is missing or empty: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(
            f"Unsupported row-id file extension {suffix!r}; use csv/tsv/txt/parquet."
        )
    if df.empty:
        raise ValueError(f"Stage 7 row-id file contains no rows: {path}")
    if "row_id" in df.columns:
        values = df["row_id"]
    elif len(df.columns) == 1:
        values = df.iloc[:, 0]
    else:
        raise ValueError(
            f"Stage 7 row-id file must contain a row_id column or one column; got {list(df.columns)}"
        )
    ids = pd.to_numeric(values, errors="raise")
    if ids.isna().any():
        raise ValueError(f"Stage 7 row-id file contains null row_id values: {path}")
    out = [int(x) for x in ids.tolist()]
    dup = sorted(pd.Series(out).value_counts()[lambda s: s > 1].index.astype(int).tolist())
    if dup:
        raise ValueError(f"Stage 7 row-id file contains duplicated row_id values: {dup[:20]}")
    return out


def _stable_group_seed(seed: int, key: object) -> int:
    payload = f"stage7_row_sample|{int(seed)}|{repr(key)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _sample_stage7_row_ids(
    *,
    panel: pd.DataFrame,
    sample_size: int,
    sample_seed: int,
    sample_strata: list[str],
) -> list[int]:
    """Deterministically sample row_ids, optionally stratified.

    Allocation is proportional to stratum size with largest-remainder residual
    allocation.  Sampling is without replacement and sorted by row_id before
    execution to keep prompt order deterministic and auditable.
    """
    n = int(sample_size)
    if n <= 0:
        raise ValueError(f"sample_size must be positive; got {sample_size!r}")
    if n > len(panel):
        raise ValueError(f"sample_size={n} exceeds available panel rows={len(panel)}")
    if "row_id" not in panel.columns:
        raise ValueError("Stage 7 sampling requires a row_id column after panel materialization.")
    if not sample_strata:
        sampled = panel.sample(n=n, random_state=int(sample_seed), replace=False)
        return sorted(pd.to_numeric(sampled["row_id"], errors="raise").astype(int).tolist())
    missing = [c for c in sample_strata if c not in panel.columns]
    if missing:
        raise ValueError(f"Stage 7 stratified sample requested missing columns: {missing}")
    work = panel.copy()
    sentinel = "__MISSING__"
    grouped = list(work.groupby(sample_strata, dropna=False, sort=True))
    alloc_rows: list[dict] = []
    for key, g in grouped:
        count = len(g)
        exact = n * count / len(work)
        base = int(exact // 1)
        alloc_rows.append({"key": key, "frame": g, "count": count, "exact": exact, "n": min(base, count)})
    allocated = sum(int(r["n"]) for r in alloc_rows)
    residual = n - allocated
    alloc_rows.sort(key=lambda r: (-(float(r["exact"]) - int(float(r["exact"]))), repr(r["key"])))
    idx = 0
    while residual > 0:
        if idx >= len(alloc_rows):
            idx = 0
        r = alloc_rows[idx]
        if int(r["n"]) < int(r["count"]):
            r["n"] = int(r["n"]) + 1
            residual -= 1
        idx += 1
    selected: list[int] = []
    for r in alloc_rows:
        take = int(r["n"])
        if take <= 0:
            continue
        g = r["frame"]
        sampled = g.sample(n=take, random_state=_stable_group_seed(sample_seed, r["key"]), replace=False)
        selected.extend(pd.to_numeric(sampled["row_id"], errors="raise").astype(int).tolist())
    selected = sorted(selected)
    if len(selected) != n:
        raise RuntimeError(f"Stage 7 sampler selected {len(selected)} rows; expected {n}.")
    return selected


def _apply_stage7_row_selection(
    *,
    panel: pd.DataFrame,
    out_dir: Path,
    row_id_file: Path | None,
    sample_size: int | None,
    sample_seed: int,
    sample_strata: list[str] | None,
) -> tuple[pd.DataFrame, dict]:
    """Apply optional N5/pilot row selection and write selected-row ledger."""
    if "row_id" not in panel.columns:
        work = panel.reset_index(drop=True).copy()
        work["row_id"] = work.index
    else:
        work = panel.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="raise").astype(int)
    duplicated_panel_ids = sorted(work["row_id"].value_counts()[lambda s: s > 1].index.astype(int).tolist())
    if duplicated_panel_ids:
        raise ValueError(f"Stage 7 serving panel contains duplicated row_id values: {duplicated_panel_ids[:20]}")
    before_count = int(len(work))
    if row_id_file is not None and sample_size is not None:
        raise ValueError("Use either --row-id-file or --sample-size, not both.")
    if row_id_file is None and sample_size is None:
        return work, {
            "enabled": False,
            "input_panel_row_count_before_selection": before_count,
            "input_panel_row_count_after_selection": before_count,
        }
    if row_id_file is not None:
        selected_ids = _read_row_id_file(Path(row_id_file))
        source = "row_id_file"
        row_file_sha = _sha256_bytes(Path(row_id_file).read_bytes())
    else:
        selected_ids = _sample_stage7_row_ids(
            panel=work,
            sample_size=int(sample_size),
            sample_seed=int(sample_seed),
            sample_strata=list(sample_strata or []),
        )
        source = "deterministic_stratified_sample" if sample_strata else "deterministic_simple_sample"
        row_file_sha = None
    panel_ids = set(work["row_id"].astype(int).tolist())
    missing = sorted(set(selected_ids) - panel_ids)
    if missing:
        raise ValueError(f"Stage 7 row selection contains row_ids absent from panel: {missing[:20]} count={len(missing)}")
    keyed = work.set_index("row_id", drop=False)
    selected = keyed.loc[selected_ids].reset_index(drop=True)
    ledger = out_dir / "llm_stage7_selected_row_ids.csv"
    pd.DataFrame({"row_id": selected_ids}).to_csv(ledger, index=False, encoding="utf-8-sig")
    selected_hash = _sha256_bytes(",".join(str(int(x)) for x in selected_ids).encode("utf-8"))
    return selected, {
        "enabled": True,
        "selection_source": source,
        "row_id_file": str(row_id_file) if row_id_file is not None else None,
        "row_id_file_sha256": row_file_sha,
        "sample_size": int(sample_size) if sample_size is not None else None,
        "sample_seed": int(sample_seed) if sample_size is not None else None,
        "sample_strata": list(sample_strata or []),
        "selected_row_ids_file": ledger.name,
        "selected_row_ids_sha256": selected_hash,
        "selected_row_count": int(len(selected_ids)),
        "input_panel_row_count_before_selection": before_count,
        "input_panel_row_count_after_selection": int(len(selected)),
    }


def _request_fingerprint(*, backend: LLMBackend, request: LLMRequest) -> str:
    """Stable identity for one LLM request under the current prompt contract.

    The fingerprint intentionally includes the structured prompt payload,
    reference value/source/seed, initial action, backend identity, mode, and
    information condition.  This prevents stale checkpoint reuse when the
    reference draw seed, C6X reference, condition matrix, or backend changes.
    """
    payload = {
        "checkpoint_schema_version": STAGE7_CHECKPOINT_SCHEMA_VERSION,
        "backend_id": backend.backend_id,
        "backend_class": type(backend).__name__,
        "row_id": int(request.row_id),
        "condition": request.condition,
        "mode": request.mode,
        "information_condition": request.information_condition,
        "prompt": request.prompt,
        "rl_reference_candidate": request.rl_reference_candidate,
        "reference_source": request.reference_source,
        "reference_draw_seed": request.reference_draw_seed,
        "initial_action": request.initial_action,
    }
    return _sha256_bytes(_json_dumps_stable(payload).encode("utf-8"))


def _parsed_action_to_checkpoint_record(
    *,
    backend: LLMBackend,
    request: LLMRequest,
    parsed: ParsedAction,
    request_seq: int,
    request_fingerprint: str,
    attempt_count: int,
) -> dict:
    rec = asdict(parsed)
    return {
        "checkpoint_schema_version": STAGE7_CHECKPOINT_SCHEMA_VERSION,
        "response_parser_grounding_matcher_version": GROUNDING_MATCHER_VERSION,
        "created_utc": _now(),
        "backend_id": backend.backend_id,
        "backend_class": type(backend).__name__,
        "request_seq": int(request_seq),
        "request_fingerprint": request_fingerprint,
        "attempt_count": int(attempt_count),
        "request_identity": {
            "row_id": int(request.row_id),
            "condition": request.condition,
            "mode": request.mode,
            "information_condition": request.information_condition,
            "rl_reference_candidate": request.rl_reference_candidate,
            "reference_source": request.reference_source,
            "reference_draw_seed": request.reference_draw_seed,
            "prompt_sha256": _sha256_bytes(_json_dumps_stable(request.prompt).encode("utf-8")),
        },
        "parsed_action": rec,
    }


def _parsed_action_from_checkpoint_record(
    rec: dict,
    *,
    request: LLMRequest | None = None,
    space=None,
) -> ParsedAction:
    """Materialize a ParsedAction from a checkpoint record.

    When the current request/space are supplied, reparse the saved raw response
    through the **current** parser rather than trusting the stored parsed_action.
    This preserves paid live responses while allowing audit-only parser fixes
    (such as grounding alias matching) to take effect on resume/rebuild runs.
    """
    payload = rec.get("parsed_action")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint record is missing parsed_action payload.")
    if request is not None and space is not None:
        raw_text = payload.get("raw_response")
        if isinstance(raw_text, str) and raw_text:
            parsed_json, parse_error = LiveLLMBackend._extract_json_block(raw_text)
            response = LLMResponse(
                request=request,
                raw_text=raw_text,
                parsed_json=parsed_json,
                parse_error=parse_error,
                backend_metadata={
                    "source": "stage7_checkpoint_reparse",
                    "grounding_matcher_version": GROUNDING_MATCHER_VERSION,
                },
            )
            return parse_response(response, space)
    return ParsedAction(**payload)


def _load_stage7_checkpoint(
    *, checkpoint_path: Path, backend: LLMBackend
) -> dict[str, tuple[int, dict]]:
    """Load usable checkpoint rows keyed by current request fingerprint.

    Malformed or stale-schema rows are ignored rather than trusted.  If a JSONL
    line is corrupted, fail fast: a partially written checkpoint must be repaired
    or removed so resume semantics stay auditable.
    """
    if not checkpoint_path.exists():
        return {}
    loaded: dict[str, tuple[int, dict]] = {}
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupt Stage 7 checkpoint JSON at {checkpoint_path}:{line_no}: {exc}"
                ) from exc
            if rec.get("checkpoint_schema_version") != STAGE7_CHECKPOINT_SCHEMA_VERSION:
                continue
            if rec.get("backend_id") != backend.backend_id:
                continue
            fp = rec.get("request_fingerprint")
            if not isinstance(fp, str) or not fp:
                continue
            # Validate the stored payload now, but keep the raw record so the
            # caller can reparse it with the current request/prompt context.
            _parsed_action_from_checkpoint_record(rec)
            seq = int(rec.get("request_seq", 0))
            loaded[fp] = (seq, rec)
    return loaded


def _append_stage7_checkpoint(
    *, checkpoint_path: Path, record: dict, lock: threading.Lock
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with lock:
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def _is_retryable_llm_response(response) -> bool:
    err = (response.parse_error or "").lower()
    if err.startswith("provider_call_failed"):
        return True
    # Live models sometimes emit empty/no-JSON responses despite JSON-only
    # instructions.  Retry these before classifying them as translational
    # failures; deterministic scripted backend normally never reaches here.
    if err in {"empty response", "no json object found", "unbalanced json braces"}:
        return True
    if err.startswith("json decode error"):
        return True
    return False


def _generate_parse_with_retry(
    *,
    backend: LLMBackend,
    request: LLMRequest,
    space,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[ParsedAction, int]:
    attempts = max(1, int(max_retries) + 1)
    last_response = None
    for attempt in range(1, attempts + 1):
        response = backend.generate(request)
        last_response = response
        if response.parse_error and _is_retryable_llm_response(response) and attempt < attempts:
            if retry_sleep_seconds > 0:
                time.sleep(float(retry_sleep_seconds) * attempt)
            continue
        if backend.is_live and response.parse_error and (response.parse_error or "").lower().startswith("provider_call_failed"):
            # Network/provider failures are not LLM decisions.  Do not silently
            # convert them into no-op actions; keep completed checkpoints and
            # let the user resume after connectivity/rate-limit recovery.
            raise RuntimeError(
                f"Live LLM provider call failed after {attempt} attempt(s) for "
                f"row_id={request.row_id}, condition={request.condition}, mode={request.mode}: "
                f"{response.parse_error}"
            )
        return parse_response(response, space), attempt
    # Defensive; loop always returns or raises.
    assert last_response is not None
    return parse_response(last_response, space), attempts


def _load_rl_reference_from_stage6(
    root: Path, final_rl_label: str
) -> dict[int, str]:
    """Read Stage 6 ``policy_actions.parquet`` and extract the RL reference
    candidate for each ``row_id``.

    Stage 6's C3 row is identified by ``policy == space.final_rl_label`` (which
    is ``"C3_candidate_iql"`` per the candidate library); the ``candidate_id``
    column for those rows is the v32 main label the RL policy selected.
    """
    s6 = stage_dir(root, "stage6") / "policy_actions.parquet"
    if not s6.exists():
        raise FileNotFoundError(
            f"Stage 7 revision conditions require Stage 6 outputs at {s6}.  "
            f"Run Stage 6 first, or restrict --conditions to C4,C5."
        )
    df = read_parquet_required(s6)
    pol = df["policy"].astype(str)
    rl_rows = df[pol == final_rl_label]
    if rl_rows.empty:
        raise ValueError(
            f"No Stage 6 rows matched policy={final_rl_label!r}.  "
            f"Cannot derive RL reference for Stage 7 revision conditions."
        )
    out = {}
    for _, r in rl_rows.iterrows():
        try:
            rid = int(r["row_id"])
        except Exception:
            continue
        out[rid] = str(r["candidate_id"])
    return out




def _stable_choice_from_labels(*, labels: list[str], row_id: int, seed: int) -> str:
    """Deterministically draw one label for C6X without relying on Python's
    process-randomized hash()."""
    if not labels:
        raise ValueError("C6X random reference draw received an empty eligible label set.")
    payload = f"c6x|{int(seed)}|{int(row_id)}".encode("utf-8")
    idx = int(hashlib.sha256(payload).hexdigest(), 16) % len(labels)
    return labels[idx]


def _build_c6x_reference_from_stage6(
    *, rl_reference: dict[int, str], space, reference_draw_seed: int
) -> dict[int, str]:
    """Build the seeded random in-vocabulary reference map for C6X.

    The draw universe is the v32 main-train vocabulary, excluding the row's
    C3/RL reference candidate when present.  This preserves source-identity
    contrast: C6 shows the RL reference; C6X shows a seed-reproducible random
    in-vocabulary reference that is not the same row's RL reference.
    """
    main_labels = list(space.train_labels)
    if not main_labels:
        raise ValueError("C6X requires non-empty space.train_labels.")
    out: dict[int, str] = {}
    for rid, rl_label in rl_reference.items():
        eligible = [x for x in main_labels if str(x) != str(rl_label)]
        out[int(rid)] = _stable_choice_from_labels(
            labels=eligible, row_id=int(rid), seed=int(reference_draw_seed)
        )
    return out




# ---------------------------------------------------------------------------
# IC-c firm-name materialization + prior-knowledge probe (LLM789-008/009)
# ---------------------------------------------------------------------------


def _norm_firm_key(x) -> str | None:
    """Mirror of Stage2 ``_norm_firm_id`` semantics for the Stage 7 boundary.

    Six-digit exchange codes with decoration (``A005930``, ``5930.0``) are
    normalized to zero-filled digit strings; digit-free values are returned
    stripped so downstream validation fails visibly instead of silently
    manufacturing a key.  Kept local (not imported from the Stage 2 pipeline)
    so Stage 7 does not acquire Stage 2's import surface; semantics must stay
    aligned with final_stage2_input_splits._norm_firm_id (RL-S2 join contract).
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s or None
    return digits.zfill(6)


def _load_firm_name_lookup(
    project_root: Path, explicit_path: Path | None
) -> tuple[dict[str, str], dict]:
    """Build firm_id -> firm_name for IC-c exposure (LLM789-008).

    Source order (each recorded in metadata; no silent fallback beyond this
    explicit ordered list):
      1. ``explicit_path`` (parquet/csv with a key column among
         {firm_id, 거래소코드, 종목코드} and a name column among
         {회사명, firm_name, 기업명}).
      2. Frozen Stage 2A panel
         ``stage2_candidate_projection/action_sources/stage2_raw_action_source_panel.parquet``
         (carries ``firm_id`` + ``회사명``).
    Hard-fails when no source yields any names.
    """
    key_cands = ["firm_id", "거래소코드", "종목코드"]
    name_cands = ["회사명", "firm_name", "기업명"]

    def _from_frame(df: pd.DataFrame, source: str, path: Path) -> tuple[dict[str, str], dict] | None:
        key = next((c for c in key_cands if c in df.columns), None)
        name = next((c for c in name_cands if c in df.columns), None)
        if key is None or name is None:
            return None
        sub = df[[key, name]].dropna()
        out: dict[str, str] = {}
        for k, v in zip(sub[key].tolist(), sub[name].tolist()):
            nk = _norm_firm_key(k)
            nv = str(v).strip()
            if nk and nv and nk not in out:
                out[nk] = nv
        if not out:
            return None
        return out, {
            "firm_name_lookup_source": source,
            "firm_name_lookup_path": str(path),
            "firm_name_lookup_key_column": key,
            "firm_name_lookup_name_column": name,
            "firm_name_lookup_n_names": int(len(out)),
        }

    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--firm-name-lookup path does not exist: {p}")
        df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        got = _from_frame(df, "explicit_path", p)
        if got is None:
            raise ValueError(
                f"--firm-name-lookup {p} lacks a usable (key, name) column pair; "
                f"need key in {key_cands} and name in {name_cands}."
            )
        return got

    s2a = (
        stage_dir(project_root, "stage2")
        / "action_sources"
        / "stage2_raw_action_source_panel.parquet"
    )
    if s2a.exists():
        got = _from_frame(pd.read_parquet(s2a), "stage2_raw_action_source_panel", s2a)
        if got is not None:
            return got
    raise FileNotFoundError(
        "IC-c firm-name exposure requires a name lookup (LLM789-008). Neither "
        f"an explicit --firm-name-lookup nor a usable Stage 2A panel at {s2a} "
        "was found. Provide --firm-name-lookup <parquet/csv with firm_id + 회사명>."
    )


def _inject_firm_names(
    panel: pd.DataFrame, lookup: dict[str, str], lookup_meta: dict
) -> tuple[pd.DataFrame, dict]:
    """Materialize ``firm_name`` onto the IC-c serving panel; hard-fail on gaps.

    Missing firm names under IC-c are not an optional-proxy situation: the
    name IS the IC-c manipulation (2026-07-04 decision), so a partially named
    panel would be a different, unauditable condition.
    """
    key_col = next((c for c in ["firm_id", "거래소코드", "종목코드"] if c in panel.columns), None)
    if key_col is None:
        raise ValueError(
            "IC-c firm-name injection requires a firm key column "
            "(firm_id/거래소코드/종목코드) on phase_eval_candidate; none found."
        )
    out = panel.copy()
    keys = out[key_col].map(_norm_firm_key)
    out["firm_name"] = keys.map(lookup)
    missing_mask = out["firm_name"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        sample = sorted({str(k) for k in keys[missing_mask].dropna().tolist()})[:20]
        raise ValueError(
            f"IC-c firm-name injection failed for {n_missing}/{len(out)} rows; "
            f"missing keys (up to 20): {sample}. Fix the lookup source "
            f"({lookup_meta.get('firm_name_lookup_source')}) or pass "
            f"--firm-name-lookup with full coverage (LLM789-008)."
        )
    meta = {
        "ic_c_firm_name_exposed": True,
        "firm_name_key_column": key_col,
        "firm_name_coverage": 1.0,
        "firm_name_rows": int(len(out)),
        **lookup_meta,
    }
    return out, meta


def _icc_probe_fingerprint(*, backend: LLMBackend, row_id: int, user_prompt: str) -> str:
    payload = {
        "probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
        "backend_id": backend.backend_id,
        "row_id": int(row_id),
        "user_prompt": user_prompt,
    }
    return _sha256_bytes(_json_dumps_stable(payload).encode("utf-8"))


def _parse_icc_probe_response(
    raw: str,
) -> tuple[float | None, bool | None, int | None, str | None]:
    """Parse a v2 numeric-recall probe response.

    Returns ``(recalled, recognized, familiarity, parse_error)``.  ``recalled``
    is the model's claimed fiscal-2024 debt-to-assets ratio (None when the
    model answered null or the field is malformed); ``recognized`` /
    ``familiarity`` are the secondary self-report fields.
    """
    parsed, err = LiveLLMBackend._extract_json_block(raw or "")
    if err or not isinstance(parsed, dict):
        return None, None, None, err or "probe response is not a JSON object"
    problems: list[str] = []
    recalled_raw = parsed.get("recalled_debt_ratio")
    recalled: float | None = None
    if recalled_raw is None:
        recalled = None
    elif isinstance(recalled_raw, bool) or not isinstance(recalled_raw, (int, float)):
        problems.append(f"recalled_debt_ratio is not numeric/null: {recalled_raw!r}")
    else:
        recalled = float(recalled_raw)
        if not (recalled == recalled) or recalled in (float("inf"), float("-inf")):
            problems.append(f"recalled_debt_ratio not finite: {recalled_raw!r}")
            recalled = None
    fam_raw = parsed.get("familiarity")
    familiarity: int | None = None
    if isinstance(fam_raw, bool) or not isinstance(fam_raw, int) or not (0 <= int(fam_raw) <= 3):
        problems.append(f"familiarity out of contract: {fam_raw!r}")
    else:
        familiarity = int(fam_raw)
    rec_raw = parsed.get("recognized")
    recognized: bool | None = None
    if isinstance(rec_raw, bool):
        recognized = bool(rec_raw)
    else:
        problems.append(f"recognized is not a bool: {rec_raw!r}")
    return recalled, recognized, familiarity, ("; ".join(problems) or None)


def _run_icc_probe_pass(
    *,
    backend: LLMBackend,
    panel: pd.DataFrame,
    information_condition: str,
    out_dir: Path,
    resume: bool,
    max_retries: int,
    retry_sleep_seconds: float,
    tolerance: float,
) -> tuple[dict[int, dict], dict]:
    """One prior-knowledge probe call per firm (IC-c only; LLM789-009).

    Sequential by design: one short identity-only prompt per firm.  Uses its
    own JSONL checkpoint (``llm_stage7_icc_probe_checkpoint.jsonl``) with the
    same resume-by-fingerprint semantics as the main pass, so paid live probe
    responses survive interruption.
    """
    if information_condition != "IC-c":
        raise ValueError("--icc-probe is defined only for --information-condition IC-c.")
    work = panel.copy()
    if "row_id" not in work.columns:
        work = work.reset_index(drop=True)
        work["row_id"] = work.index
    if work["row_id"].duplicated().any():
        dup = sorted(work.loc[work["row_id"].duplicated(), "row_id"].astype(int).unique().tolist())[:10]
        raise ValueError(f"IC-c probe requires unique row_id per firm; duplicates: {dup}")
    if not (float(tolerance) > 0.0):
        raise ValueError(f"--icc-probe-tolerance must be > 0; got {tolerance!r}")
    if ICC_PROBE_TARGET_FEATURE not in work.columns:
        raise ValueError(
            f"IC-c probe v2 compares recalled values against panel column "
            f"{ICC_PROBE_TARGET_FEATURE!r}, which is missing from "
            f"phase_eval_candidate. The probe cannot run without its offline "
            f"comparison target (LLM789-009 v2)."
        )

    ckpt = out_dir / ICC_PROBE_CHECKPOINT_NAME
    cached: dict[str, dict] = {}
    if resume and ckpt.exists():
        with ckpt.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt IC-c probe checkpoint at {ckpt}:{line_no}: {exc}") from exc
                if rec.get("probe_schema_version") != ICC_PROBE_SCHEMA_VERSION:
                    continue
                if rec.get("backend_id") != backend.backend_id:
                    continue
                fp = rec.get("probe_fingerprint")
                if isinstance(fp, str) and fp:
                    cached[fp] = rec

    probe_map: dict[int, dict] = {}
    resumed = 0
    parse_failures = 0
    recognized_count = 0
    contaminated_count = 0
    recall_null_count = 0
    panel_missing_count = 0
    attempts_total = max(1, int(max_retries) + 1)
    for _, row in work.iterrows():
        rid = int(row["row_id"])
        prompt = build_icc_probe_prompt(row, information_condition)
        fp = _icc_probe_fingerprint(backend=backend, row_id=rid, user_prompt=prompt["user_prompt"])
        rec = cached.get(fp)
        if rec is None:
            raw = None
            last_exc: Exception | None = None
            for attempt in range(1, attempts_total + 1):
                try:
                    raw = backend.generate_raw(prompt["system_prompt"], prompt["user_prompt"])
                    break
                except Exception as exc:  # provider/transport failure
                    last_exc = exc
                    if attempt < attempts_total and retry_sleep_seconds > 0:
                        time.sleep(float(retry_sleep_seconds) * attempt)
            if raw is None:
                # Mirror the main-pass policy: provider failures on a live
                # backend abort with checkpoints intact rather than silently
                # degrading probe coverage.
                raise RuntimeError(
                    f"IC-c probe provider call failed after {attempts_total} attempt(s) "
                    f"for row_id={rid}: {type(last_exc).__name__}: {last_exc}"
                )
            recalled, recognized, familiarity, perr = _parse_icc_probe_response(raw)
            panel_raw = row.get(ICC_PROBE_TARGET_FEATURE)
            panel_v = None if pd.isna(panel_raw) else float(panel_raw)
            rel_err = None
            if recalled is not None and panel_v is not None:
                rel_err = abs(recalled - panel_v) / max(abs(panel_v), 1e-9)
            # NOTE: the contamination flag is NOT stored in the checkpoint —
            # it is recomputed from rel_err against the CURRENT tolerance on
            # every load, so a tolerance change never silently reuses stale
            # flags from a prior run.
            rec = {
                "probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
                "created_utc": _now(),
                "backend_id": backend.backend_id,
                "probe_fingerprint": fp,
                "row_id": rid,
                "icc_probe_response_raw": raw,
                "icc_probe_value": recalled,
                "icc_probe_parse_error": perr,
                "icc_probe_rel_err": rel_err,
                "icc_probe_panel_value": panel_v,
                "icc_probe_recognized": recognized,
                "icc_probe_familiarity": familiarity,
            }
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with ckpt.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
        else:
            resumed += 1
        if rec.get("icc_probe_parse_error"):
            parse_failures += 1
        rel_err = rec.get("icc_probe_rel_err")
        flag = (float(rel_err) <= float(tolerance)) if rel_err is not None else None
        if flag is True:
            contaminated_count += 1
        if rec.get("icc_probe_value") is None and not rec.get("icc_probe_parse_error"):
            recall_null_count += 1
        if rec.get("icc_probe_panel_value") is None:
            panel_missing_count += 1
        if rec.get("icc_probe_recognized") is True:
            recognized_count += 1
        probe_map[rid] = {
            "icc_probe_response_raw": rec.get("icc_probe_response_raw"),
            "icc_probe_value": rec.get("icc_probe_value"),
            "icc_contamination_flag": flag,
            "icc_probe_parse_error": rec.get("icc_probe_parse_error"),
            "icc_probe_rel_err": rel_err,
            "icc_probe_panel_value": rec.get("icc_probe_panel_value"),
        }
    meta = {
        "icc_probe_enabled": True,
        "icc_probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
        "icc_probe_target_feature": ICC_PROBE_TARGET_FEATURE,
        "icc_probe_tolerance_relative": float(tolerance),
        "icc_probe_checkpoint": str(ckpt),
        "icc_probe_row_count": int(len(probe_map)),
        "icc_probe_resumed_count": int(resumed),
        "icc_probe_parse_failures": int(parse_failures),
        "icc_probe_contaminated_count": int(contaminated_count),
        "icc_probe_recall_null_count": int(recall_null_count),
        "icc_probe_panel_value_missing_count": int(panel_missing_count),
        "icc_probe_recognized_count": int(recognized_count),
    }
    return probe_map, meta


def _join_icc_probe_columns(df: pd.DataFrame, probe_map: dict[int, dict] | None) -> pd.DataFrame:
    """Attach the contract-v4 icc_probe_* columns (always present for schema
    stability; all-null when the probe is disabled)."""
    out = df.copy()
    if out.empty:
        for c in ICC_PROBE_COLUMNS:
            out[c] = pd.Series(dtype="object")
        return out
    if probe_map is None:
        for c in ICC_PROBE_COLUMNS:
            out[c] = None
        return out
    rid = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    for c in ICC_PROBE_COLUMNS:
        out[c] = rid.map(lambda r: probe_map.get(int(r), {}).get(c)).astype("object")
    return out


def _materialize_initial_action(parsed: ParsedAction) -> dict:
    """Stage 7 revision conditions C4R/C6/C6X/C7 need ``a_0`` — the initial action
    produced by C4/C5 — recorded into the self-revision/reference-revision prompt."""
    return {
        "selected_candidate": parsed.selected_candidate,
        "materialized_action": dict(parsed.materialized_action),
        "rationale": parsed.rationale,
    }


def _run_condition_pass(
    *,
    backend: LLMBackend,
    base_panel: pd.DataFrame,
    space,
    conditions: list[str],
    modes: list[str],
    information_condition: str,
    rl_reference: dict[int, str] | None,
    c6x_reference: dict[int, str] | None,
    reference_draw_seed: int | None,
    initial_actions_by_key: dict[tuple[int, str, str], ParsedAction] | None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    max_concurrency: int = 1,
    max_retries: int = 2,
    retry_sleep_seconds: float = 1.0,
    action_budget_contract: dict | None = None,
    prompt_payload_records: list[dict] | None = None,
) -> list[ParsedAction]:
    """Run one pass over (rows × conditions × modes) with checkpoint/resume.

    ``initial_actions_by_key`` keys are ``(row_id, base_condition, mode)``
    where ``base_condition`` is the C4/C5 source for the C4R/C6/C6X/C7 revision.

    Completed requests are appended to a JSONL checkpoint immediately after
    parsing, before final Stage 7 artifacts are materialized.  On rerun, the
    same request fingerprint is skipped and reconstructed from checkpoint.
    Bounded concurrency is applied only within a pass; the first-pass →
    revision-pass dependency remains sequential so C4R/C6/C6X/C7 always receive
    their matching initial action.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1; got {max_concurrency}.")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0; got {max_retries}.")
    if retry_sleep_seconds < 0:
        raise ValueError(f"retry_sleep_seconds must be >= 0; got {retry_sleep_seconds}.")

    if "row_id" not in base_panel.columns:
        work = base_panel.reset_index(drop=True).copy()
        work["row_id"] = work.index
    else:
        work = base_panel.copy()

    request_items: list[tuple[int, LLMRequest, str]] = []
    seq = 0
    for _, row in work.iterrows():
        rid = int(row["row_id"])
        rl_ref = rl_reference.get(rid) if rl_reference is not None else None
        c6x_ref = c6x_reference.get(rid) if c6x_reference is not None else None
        for cond in conditions:
            if cond == "C6X":
                ref_for_cond = c6x_ref
                reference_source = "random"
                ref_seed_for_cond = int(reference_draw_seed) if reference_draw_seed is not None else None
            elif cond in {"C6", "C7", "C8"}:
                ref_for_cond = rl_ref
                reference_source = "rl"
                ref_seed_for_cond = None
            else:
                ref_for_cond = None
                reference_source = "none"
                ref_seed_for_cond = None
            if cond in {"C6", "C6X", "C7", "C8"} and ref_for_cond is None:
                raise ValueError(
                    f"Missing reference candidate for row_id={rid} required by {cond}."
                )
            for mode in modes:
                initial_action = None
                if cond in {"C4R", "C6", "C6X", "C7"} and initial_actions_by_key is not None:
                    base_cond = "C4" if cond in {"C4R", "C6", "C6X"} else "C5"
                    key = (rid, base_cond, mode)
                    init = initial_actions_by_key.get(key)
                    if init is None:
                        raise ValueError(
                            f"Stage 7 second pass missing initial action for "
                            f"(row_id={rid}, base={base_cond}, mode={mode})."
                        )
                    initial_action = _materialize_initial_action(init)

                prompt_payload = build_prompt(
                    row=row,
                    condition=cond,
                    mode=mode,
                    information_condition=information_condition,
                    space=space,
                    rl_reference_candidate=ref_for_cond,
                    reference_source=reference_source,
                    reference_draw_seed=ref_seed_for_cond,
                    initial_action=initial_action,
                    action_budget_contract=action_budget_contract,
                )
                request = LLMRequest(
                    row_id=rid,
                    condition=cond,
                    mode=mode,
                    information_condition=information_condition,
                    prompt=prompt_payload,
                    rl_reference_candidate=ref_for_cond,
                    reference_source=reference_source,
                    reference_draw_seed=ref_seed_for_cond,
                    initial_action=initial_action,
                )
                fp = _request_fingerprint(backend=backend, request=request)
                if prompt_payload_records is not None:
                    prompt_text = _json_dumps_stable(request.prompt)
                    prompt_payload_records.append({
                        "schema_version": STAGE7_PROMPT_PAYLOAD_ARCHIVE_SCHEMA_VERSION,
                        "request_seq_within_pass": int(seq),
                        "request_fingerprint": fp,
                        "backend_id": backend.backend_id,
                        "backend_class": type(backend).__name__,
                        "row_id": int(request.row_id),
                        "condition": request.condition,
                        "mode": request.mode,
                        "information_condition": request.information_condition,
                        "rl_reference_candidate": request.rl_reference_candidate,
                        "reference_source": request.reference_source,
                        "reference_draw_seed": request.reference_draw_seed,
                        "initial_action": request.initial_action,
                        "prompt_payload": request.prompt,
                        "prompt_sha256": _sha256_bytes(prompt_text.encode("utf-8")),
                    })
                request_items.append((seq, request, fp))
                seq += 1

    checkpoint_records: dict[str, tuple[int, dict]] = {}
    if checkpoint_path is not None and resume:
        checkpoint_records = _load_stage7_checkpoint(
            checkpoint_path=checkpoint_path, backend=backend
        )

    parsed_by_seq: dict[int, ParsedAction] = {}
    pending: list[tuple[int, LLMRequest, str]] = []
    for request_seq, request, fp in request_items:
        cached = checkpoint_records.get(fp)
        if cached is not None:
            _, rec = cached
            parsed_by_seq[request_seq] = _parsed_action_from_checkpoint_record(
                rec, request=request, space=space
            )
        else:
            pending.append((request_seq, request, fp))

    checkpoint_lock = threading.Lock()

    def _run_one(item: tuple[int, LLMRequest, str]) -> tuple[int, ParsedAction]:
        request_seq, request, fp = item
        parsed, attempt_count = _generate_parse_with_retry(
            backend=backend,
            request=request,
            space=space,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if checkpoint_path is not None:
            rec = _parsed_action_to_checkpoint_record(
                backend=backend,
                request=request,
                parsed=parsed,
                request_seq=request_seq,
                request_fingerprint=fp,
                attempt_count=attempt_count,
            )
            _append_stage7_checkpoint(
                checkpoint_path=checkpoint_path,
                record=rec,
                lock=checkpoint_lock,
            )
        return request_seq, parsed

    if pending:
        if int(max_concurrency) == 1:
            for item in pending:
                request_seq, parsed = _run_one(item)
                parsed_by_seq[request_seq] = parsed
        else:
            with ThreadPoolExecutor(max_workers=int(max_concurrency)) as executor:
                futures = [executor.submit(_run_one, item) for item in pending]
                for fut in as_completed(futures):
                    request_seq, parsed = fut.result()
                    parsed_by_seq[request_seq] = parsed

    missing = sorted(set(range(len(request_items))) - set(parsed_by_seq))
    if missing:
        raise RuntimeError(f"Stage 7 internal error: missing parsed requests for seq={missing[:10]}.")
    return [parsed_by_seq[i] for i in range(len(request_items))]


def run_stage7(
    *,
    project_root: Path,
    backend_spec: str,
    information_condition: str = "IC-a",
    conditions: list[str] | None = None,
    modes: list[str] | None = None,
    allow_no_rl_reference: bool = False,
    reference_draw_seed: int | None = None,
    backend_kwargs: dict | None = None,
    max_concurrency: int = 1,
    resume: bool = True,
    checkpoint_path: Path | None = None,
    max_retries: int = 2,
    retry_sleep_seconds: float = 1.0,
    icc_probe: bool = False,
    firm_name_lookup_path: Path | None = None,
    icc_probe_tolerance: float = 0.20,
    candidate_library_quantile: int | None = 50,
    row_id_file: Path | None = None,
    sample_size: int | None = None,
    sample_seed: int = 20260707,
    sample_strata: list[str] | None = None,
    freeform_l1_budget: float | None = None,
    budgeted_conditions: list[str] | None = None,
    budget_contract_label: str | None = None,
    budget_tolerance: float = 1.0e-9,
) -> dict:
    """Execute Stage 7 end-to-end and write all outputs.

    Returns the metadata dict that is also written to ``metadata.json``.
    """
    project_root = Path(project_root).resolve()
    final = final_root(project_root)
    out_dir = stage_dir(project_root, "stage7")
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        checkpoint_path = out_dir / DEFAULT_STAGE7_CHECKPOINT_NAME
    else:
        checkpoint_path = Path(checkpoint_path)

    if conditions is None:
        conditions = ["C4", "C5", "C6", "C6X", "C7", "C8"]
    if modes is None:
        modes = ["candidate_selection", "free_form_10d"]

    for c in conditions:
        if c not in {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}:
            raise ValueError(f"Unsupported Stage 7 condition: {c}")
    for m in modes:
        if m not in {"candidate_selection", "free_form_10d"}:
            raise ValueError(f"Unsupported Stage 7 mode: {m}")
    if information_condition not in {"IC-a", "IC-b", "IC-c"}:
        raise ValueError(f"Unsupported information condition: {information_condition}")
    if icc_probe and information_condition != "IC-c":
        raise ValueError("--icc-probe is defined only for --information-condition IC-c (LLM789-009).")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1; got {max_concurrency}.")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0; got {max_retries}.")
    if retry_sleep_seconds < 0:
        raise ValueError(f"retry_sleep_seconds must be >= 0; got {retry_sleep_seconds}.")

    action_budget_contract = make_action_budget_contract(
        l1_budget=freeform_l1_budget,
        budgeted_conditions=budgeted_conditions,
        label=budget_contract_label,
        budgeted_modes=["free_form_10d"],
        tolerance=float(budget_tolerance),
    )
    action_budget_contract_meta = (
        action_budget_contract.to_dict() if action_budget_contract is not None else disabled_budget_contract()
    )
    if action_budget_contract is not None:
        if "free_form_10d" not in modes:
            raise ValueError("--freeform-l1-budget requires --modes to include free_form_10d.")
        active_budgeted = sorted(set(action_budget_contract.budgeted_conditions).intersection(set(conditions)))
        if not active_budgeted:
            raise ValueError(
                "--freeform-l1-budget was supplied but none of the budgeted conditions "
                f"{list(action_budget_contract.budgeted_conditions)} are present in --conditions={conditions}."
            )

    # --- load contract artifacts ---
    # LLM action materialization/projection must use the same Stage2-selected
    # magnitude-calibrated candidate vectors as the RL track (P50 by default).
    # The active base YAML remains immutable for config-freeze/hash provenance.
    base_hashes = active_config_hashes(project_root)
    selected_candidate_library_path = resolve_candidate_library_path(
        project_root, magnitude_quantile=candidate_library_quantile
    )
    space = load_action_space(project_root, candidate_library_path=selected_candidate_library_path)
    hashes = dict(base_hashes)
    hashes["selected_candidate_library_hash"] = space.candidate_library_hash
    hashes["selected_candidate_library_path"] = str(selected_candidate_library_path)

    # --- load state panel from Stage 2 (state-only) ---
    state_panel_path = stage_dir(project_root, "stage2") / "phase_eval_candidate.parquet"
    if not state_panel_path.exists():
        raise FileNotFoundError(
            f"Stage 7 requires Stage 2 phase_eval_candidate.parquet at {state_panel_path}."
        )
    panel = read_parquet_required(state_panel_path)
    # phase_eval_candidate is state-only by design; refuse to proceed if a
    # forbidden column has leaked.  This is a fail-fast on the upstream
    # contract.
    forbidden_present = [c for c in panel.columns if str(c).startswith("next__") or str(c).startswith("action__") or str(c) in {"reward_train", "reward_raw"}]
    if forbidden_present:
        raise ValueError(
            f"Stage 7 refuses to consume phase_eval_candidate with forbidden "
            f"columns: {forbidden_present}.  Re-run Stage 2 with the "
            f"phase_eval state-only contract."
        )

    panel, row_selection_meta = _apply_stage7_row_selection(
        panel=panel,
        out_dir=out_dir,
        row_id_file=row_id_file,
        sample_size=sample_size,
        sample_seed=int(sample_seed),
        sample_strata=list(sample_strata or []),
    )

    # --- IC-c firm-name exposure (LLM789-008; research decision 2026-07-04) ---
    firm_name_meta: dict = {"ic_c_firm_name_exposed": False}
    if information_condition == "IC-c":
        lookup, lookup_meta = _load_firm_name_lookup(project_root, firm_name_lookup_path)
        panel, firm_name_meta = _inject_firm_names(panel, lookup, lookup_meta)

    # --- RL reference for revision conditions ---
    needs_rl_ref = any(c in {"C6", "C6X", "C7", "C8"} for c in conditions)
    rl_reference: dict[int, str] | None = None
    if needs_rl_ref:
        if allow_no_rl_reference:
            rl_reference = {}
        else:
            rl_reference = _load_rl_reference_from_stage6(
                project_root, final_rl_label=space.final_rl_label
            )

    if reference_draw_seed is None:
        reference_draw_seed = 20260523
    c6x_reference: dict[int, str] | None = None
    if "C6X" in conditions:
        if rl_reference is None:
            raise ValueError("C6X requires the Stage 6 RL reference map before drawing random references.")
        c6x_reference = _build_c6x_reference_from_stage6(
            rl_reference=rl_reference, space=space, reference_draw_seed=int(reference_draw_seed)
        )

    # --- instantiate backend ---
    backend_kwargs = backend_kwargs or {}
    backend = make_backend(backend_spec, **backend_kwargs)

    # Full prompt payload archive for future runs. Existing frozen archives remain hash-only.
    prompt_payload_records: list[dict] = []

    # --- first pass: non-revision conditions (C4, C5) + C8 ---
    first_pass_conditions = [c for c in conditions if c in {"C4", "C5", "C8"}]
    revision_conditions = [c for c in conditions if c in {"C4R", "C6", "C6X", "C7"}]

    parsed_first = _run_condition_pass(
        backend=backend,
        base_panel=panel,
        space=space,
        conditions=first_pass_conditions,
        modes=modes,
        information_condition=information_condition,
        rl_reference=rl_reference,
        c6x_reference=c6x_reference,
        reference_draw_seed=int(reference_draw_seed),
        initial_actions_by_key=None,
        checkpoint_path=checkpoint_path,
        resume=bool(resume),
        max_concurrency=int(max_concurrency),
        max_retries=int(max_retries),
        retry_sleep_seconds=float(retry_sleep_seconds),
        action_budget_contract=action_budget_contract_meta,
        prompt_payload_records=prompt_payload_records,
    )

    # --- second pass: C4R/C6/C7 with initial actions from C4/C5 ---
    initial_actions_by_key: dict[tuple[int, str, str], ParsedAction] = {}
    for p in parsed_first:
        if p.policy in {"C4", "C5"}:
            initial_actions_by_key[(p.row_id, p.policy, p.mode)] = p

    parsed_second: list[ParsedAction] = []
    if revision_conditions:
        missing_bases = []
        for cond in revision_conditions:
            base = "C4" if cond in {"C4R", "C6", "C6X"} else "C5"
            if base not in conditions:
                missing_bases.append(f"{cond} requires {base} in --conditions")
        if missing_bases:
            raise ValueError(
                "Stage 7 revision conditions need their base condition: "
                + "; ".join(missing_bases)
            )
        parsed_second = _run_condition_pass(
            backend=backend,
            base_panel=panel,
            space=space,
            conditions=revision_conditions,
            modes=modes,
            information_condition=information_condition,
            rl_reference=rl_reference,
            c6x_reference=c6x_reference,
            reference_draw_seed=int(reference_draw_seed),
            initial_actions_by_key=initial_actions_by_key,
            checkpoint_path=checkpoint_path,
            resume=bool(resume),
            max_concurrency=int(max_concurrency),
            max_retries=int(max_retries),
            retry_sleep_seconds=float(retry_sleep_seconds),
            action_budget_contract=action_budget_contract_meta,
            prompt_payload_records=prompt_payload_records,
        )

    all_parsed = parsed_first + parsed_second

    # --- project free-form actions ---
    project_free_form_batch(all_parsed, space)

    # --- IC-c prior-knowledge probe pass (LLM789-009) ---
    probe_map: dict[int, dict] | None = None
    icc_probe_meta: dict = {"icc_probe_enabled": False}
    if icc_probe:
        probe_map, icc_probe_meta = _run_icc_probe_pass(
            backend=backend,
            panel=panel,
            information_condition=information_condition,
            out_dir=out_dir,
            resume=bool(resume),
            max_retries=int(max_retries),
            retry_sleep_seconds=float(retry_sleep_seconds),
            tolerance=float(icc_probe_tolerance),
        )

    # --- write outputs ---
    action_table = to_policy_actions_frame(all_parsed, space)
    if not action_table.empty:
        # Fail-fast: action column order must equal final_action_contract.
        action_cols_in_frame = [c for c in action_table.columns if c.startswith("action__")]
        if action_cols_in_frame != list(space.columns):
            raise ValueError(
                f"Stage 7 action column order differs from final_action_contract: "
                f"got {action_cols_in_frame}, expected {space.columns}"
            )
        # Fail-fast: every selected_candidate must be in v32 main labels.
        bad = sorted(set(action_table["candidate_id"].astype(str)) - set(space.train_labels))
        if bad:
            raise ValueError(
                f"Stage 7 produced candidates outside v32 main_train_labels: {bad}"
            )
    action_table = _join_icc_probe_columns(action_table, probe_map)
    action_table.to_parquet(out_dir / "llm_stage7_action_table.parquet", index=False)

    failure_audit = to_failure_audit_frame(all_parsed)
    failure_audit = _join_icc_probe_columns(failure_audit, probe_map)
    failure_audit.to_csv(out_dir / "llm_stage7_failure_audit.csv", index=False, encoding="utf-8-sig")

    response_rows = []
    prompt_records = []
    for p in all_parsed:
        response_rows.append({
            "row_id": p.row_id,
            "policy": p.policy,
            "mode": p.mode,
            "information_condition": p.information_condition,
            "raw_response_sha256": _sha256_bytes((p.raw_response or "").encode("utf-8")),
            "raw_response": p.raw_response,
        })
    pd.DataFrame(response_rows).to_parquet(
        out_dir / "llm_stage7_response_log.parquet", index=False
    )

    if len(prompt_payload_records) != len(all_parsed):
        raise RuntimeError(
            f"Stage 7 prompt payload count mismatch: prompts={len(prompt_payload_records)}, parsed={len(all_parsed)}"
        )
    prompt_payload_path = out_dir / "llm_stage7_prompt_payloads.jsonl"
    prompt_payload_path.write_text(
        "".join(_json_dumps_stable(record) + "\n" for record in prompt_payload_records),
        encoding="utf-8",
    )

    # Compact prompt summary; full structured payloads are in prompt_payload_path.
    for p in all_parsed:
        prompt_records.append({
            "row_id": p.row_id,
            "condition": p.policy,
            "mode": p.mode,
            "information_condition": p.information_condition,
            "rl_reference_candidate": p.rl_reference_candidate,
            "reference_source": p.reference_source,
            "reference_draw_seed": p.reference_draw_seed,
            "budget_contract_label": p.budget_contract_label,
            "budgeted_condition_flag": bool(p.budgeted_condition_flag),
            "budget_l1_target": p.budget_l1_target,
            "budget_l1_raw": p.budget_l1_raw,
            "budget_l1_clipped": p.budget_l1_clipped,
            "budget_compliant_raw": p.budget_compliant_raw,
            "budget_compliant_clipped": p.budget_compliant_clipped,
            "selected_candidate": p.selected_candidate,
            "rationale_sha256": _sha256_bytes((p.rationale or "").encode("utf-8")),
            "raw_response_sha256": _sha256_bytes((p.raw_response or "").encode("utf-8")),
        })
    manifest = {
        "stage": "final_stage7_llm_action_generation",
        "created_utc": _now(),
        "backend": backend.manifest(),
        "checkpoint": {
            "enabled": checkpoint_path is not None,
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "schema_version": STAGE7_CHECKPOINT_SCHEMA_VERSION,
            "grounding_matcher_version": GROUNDING_MATCHER_VERSION,
            "resume": bool(resume),
            "max_concurrency": int(max_concurrency),
            "max_retries": int(max_retries),
            "retry_sleep_seconds": float(retry_sleep_seconds),
        },
        "information_condition": information_condition,
        "conditions": conditions,
        "modes": modes,
        "reference_draw_seed": int(reference_draw_seed),
        "response_parser_grounding_matcher_version": GROUNDING_MATCHER_VERSION,
        "c6x_random_reference_count": int(len(c6x_reference or {})),
        "reference_source_policy": {"C4R": "none", "C6": "rl", "C6X": "random", "C7": "rl", "C8": "rl"},
        "action_budget_contract": action_budget_contract_meta,
        "row_selection_contract": row_selection_meta,
        "candidate_action_values_source": "stage2_recalibrated_candidate_library",
        "candidate_library_quantile": int(candidate_library_quantile) if candidate_library_quantile is not None else None,
        "candidate_library_path": str(selected_candidate_library_path),
        "candidate_library_hash": space.candidate_library_hash,
        "base_candidate_library_path": base_hashes["candidate_library_path"],
        "base_candidate_library_hash": base_hashes["candidate_library_hash"],
        "row_count": int(action_table["row_id"].nunique()) if not action_table.empty else 0,
        "request_count": len(all_parsed),
        "routed_to_simulator_count": int(sum(1 for p in all_parsed if p.routed_to_simulator)),
        "failure_count_by_category": {
            cat: int(sum(1 for p in all_parsed if cat in p.failure_categories))
            for cat in {
                "translational_failure", "structural_out_of_scope",
                "direction_error", "magnitude_error",
                "feasibility_violation", "liquidity_destructive_recourse",
                "anchoring_or_confirmation_failure", "ungrounded_judgment",
            }
        },
        "stage7_contract_addendum": STAGE7_CONTRACT_ADDENDUM,
        "ic_c_firm_name_exposed": bool(firm_name_meta.get("ic_c_firm_name_exposed")),
        "icc_probe_enabled": bool(icc_probe_meta.get("icc_probe_enabled")),
        "prompt_payload_archive": {
            "schema_version": STAGE7_PROMPT_PAYLOAD_ARCHIVE_SCHEMA_VERSION,
            "path": prompt_payload_path.name,
            "sha256": _sha256_bytes(prompt_payload_path.read_bytes()),
            "record_count": len(prompt_payload_records),
            "preservation_policy": "full_structured_prompt_payload_per_request",
        },
        "prompts": prompt_records,
    }
    write_json(out_dir / "llm_stage7_prompt_manifest.json", manifest)

    # --- metadata ---
    final_paper_run_allowed = backend.is_live and bool(rl_reference) if needs_rl_ref else backend.is_live
    meta = {
        "stage": "final_stage7_llm_action_generation",
        "status": "PASS" if backend.is_live else "PASS_REPRODUCIBILITY_BACKEND",
        "created_utc": _now(),
        "backend_id": backend.backend_id,
        "backend_class": type(backend).__name__,
        "backend_is_live": bool(backend.is_live),
        "final_paper_run_allowed": bool(final_paper_run_allowed),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_schema_version": STAGE7_CHECKPOINT_SCHEMA_VERSION,
        "response_parser_grounding_matcher_version": GROUNDING_MATCHER_VERSION,
        "resume_enabled": bool(resume),
        "max_concurrency": int(max_concurrency),
        "max_retries": int(max_retries),
        "retry_sleep_seconds": float(retry_sleep_seconds),
        "information_condition": information_condition,
        "conditions": conditions,
        "modes": modes,
        "reference_draw_seed": int(reference_draw_seed),
        "c6x_random_reference_count": int(len(c6x_reference or {})),
        "reference_source_policy": {"C4R": "none", "C6": "rl", "C6X": "random", "C7": "rl", "C8": "rl"},
        "action_budget_contract": action_budget_contract_meta,
        "action_budget_contract_schema_version": ACTION_BUDGET_CONTRACT_SCHEMA_VERSION,
        "row_selection_contract": row_selection_meta,
        "row_count": int(action_table["row_id"].nunique()) if not action_table.empty else 0,
        "request_count": len(all_parsed),
        "routed_to_simulator_count": int(sum(1 for p in all_parsed if p.routed_to_simulator)),
        "candidate_library_hash": space.candidate_library_hash,
        "candidate_library_path": str(selected_candidate_library_path),
        "candidate_action_values_source": "stage2_recalibrated_candidate_library",
        "candidate_library_quantile": int(candidate_library_quantile) if candidate_library_quantile is not None else None,
        "selected_recalibrated_candidate_library_hash": space.candidate_library_hash,
        "selected_recalibrated_candidate_library_path": str(selected_candidate_library_path),
        "base_candidate_library_hash": base_hashes["candidate_library_hash"],
        "base_candidate_library_path": base_hashes["candidate_library_path"],
        "final_action_contract_hash": hashes["final_action_contract_hash"],
        "stage6_policy_actions_consumed": needs_rl_ref,
        "phase_eval_candidate_source": str(state_panel_path),
        "stage7_contract_addendum": STAGE7_CONTRACT_ADDENDUM,
        "prompt_payload_archive_contract": {
            "schema_version": STAGE7_PROMPT_PAYLOAD_ARCHIVE_SCHEMA_VERSION,
            "status": "FULL_PAYLOAD_ARCHIVED",
            "path": prompt_payload_path.name,
            "sha256": _sha256_bytes(prompt_payload_path.read_bytes()),
            "record_count": len(prompt_payload_records),
        },
        **firm_name_meta,
        **icc_probe_meta,
        "outputs": {
            "action_table": "llm_stage7_action_table.parquet",
            "prompt_manifest": "llm_stage7_prompt_manifest.json",
            "prompt_payloads": "llm_stage7_prompt_payloads.jsonl",
            "failure_audit": "llm_stage7_failure_audit.csv",
            "response_log": "llm_stage7_response_log.parquet",
            "selected_row_ids": "llm_stage7_selected_row_ids.csv" if row_selection_meta.get("enabled") else None,
        },
    }
    write_json(out_dir / "metadata.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 7 — LLM Recourse Action Generation")
    ap.add_argument("--project-root", required=True)
    ap.add_argument(
        "--backend",
        default="scripted",
        help="Backend spec: 'scripted', 'openai:<model>', 'anthropic:<model>', or 'gemini:<model>'.",
    )
    ap.add_argument(
        "--information-condition",
        default="IC-a",
        choices=["IC-a", "IC-b", "IC-c"],
    )
    ap.add_argument(
        "--conditions",
        default="C4,C5,C6,C6X,C7,C8",
        help="Comma-separated subset of {C4,C4R,C5,C6,C6X,C7,C8}.",
    )
    ap.add_argument(
        "--modes",
        default="candidate_selection,free_form_10d",
        help="Comma-separated subset of {candidate_selection, free_form_10d}.",
    )
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--reference-draw-seed", type=int, default=None)
    ap.add_argument(
        "--candidate-library-quantile",
        type=int,
        default=50,
        choices=[50, 65, 75, 85],
        help="Stage2 materialized candidate-library quantile for LLM prompt/projection/simulation vectors; default P50 aligns LLM with RL.",
    )
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument(
        "--openai-api-mode",
        choices=["chat", "responses"],
        default=None,
        help=(
            "OpenAI API mode.  Default is legacy chat unless "
            "--openai-reasoning-effort is supplied, in which case the "
            "OpenAI backend uses Responses API."
        ),
    )
    ap.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Enable OpenAI Responses API reasoning with the selected effort.",
    )
    ap.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=None,
        help="Optional Responses API max_output_tokens budget for reasoning/output tokens.",
    )
    ap.add_argument("--anthropic-max-tokens", type=int, default=None,
                    help="Anthropic max_tokens budget; default preserves the backend default (1024).")
    ap.add_argument("--anthropic-temperature", type=float, default=None,
                    help="Anthropic sampling temperature; omit when extended thinking is enabled.")
    ap.add_argument("--anthropic-thinking-budget-tokens", type=int, default=None,
                    help="Enable Anthropic manual extended thinking with this token budget; must be >=1024 and < --anthropic-max-tokens.")
    ap.add_argument(
        "--gemini-thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Gemini 3.x thinking level; journal amendment uses low.",
    )
    ap.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=None,
        help="Gemini generateContent maxOutputTokens.",
    )
    ap.add_argument(
        "--gemini-response-mime-type",
        choices=["application/json", "text/plain"],
        default=None,
        help="Gemini generateContent response MIME type.",
    )
    ap.add_argument(
        "--gemini-timeout-seconds",
        type=float,
        default=None,
        help="Per-request Gemini HTTP timeout in seconds.",
    )
    ap.add_argument("--icc-probe", action="store_true",
                    help="Run the IC-c prior-knowledge probe (LLM789-009). IC-c only; default off.")
    ap.add_argument("--firm-name-lookup", default=None,
                    help="Explicit firm_id→회사명 lookup (parquet/csv) for IC-c firm-name exposure (LLM789-008).")
    ap.add_argument("--icc-probe-tolerance", type=float, default=0.20,
                    help="Pre-registered relative tolerance for the v2 numeric-recall contamination flag (LLM789-009 v2). Default 0.20.")
    ap.add_argument("--row-id-file", default=None,
                    help="Optional pre-registered row_id subset file (csv/tsv/txt/parquet) for pilot runs such as N5.")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="Optional deterministic row sample size when no --row-id-file is provided.")
    ap.add_argument("--sample-seed", type=int, default=20260707,
                    help="Seed for deterministic Stage7 row sampling; default 20260707.")
    ap.add_argument("--sample-strata", default=None,
                    help="Comma-separated panel columns for proportional stratified sampling, e.g. sector_7,grade_base_10.")
    ap.add_argument("--freeform-l1-budget", type=float, default=None,
                    help="N5-style generation-time L1 budget for free_form_10d action vectors.")
    ap.add_argument("--budgeted-conditions", default="C6",
                    help="Comma-separated policy conditions receiving --freeform-l1-budget; default C6.")
    ap.add_argument("--budget-contract-label", default=None,
                    help="Directory-safe label recorded in prompt/metadata/audit for the budget contract.")
    ap.add_argument("--budget-tolerance", type=float, default=1.0e-9,
                    help="Numerical tolerance for budget-compliance audit; default 1e-9.")
    args = ap.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sample_strata = [c.strip() for c in (args.sample_strata or "").split(",") if c.strip()]
    budgeted_conditions = [c.strip() for c in (args.budgeted_conditions or "").split(",") if c.strip()]
    backend_kwargs: dict = {}
    if args.backend == "scripted":
        backend_kwargs["seed"] = args.seed
    elif args.backend.strip().lower().startswith("openai:"):
        if args.openai_api_mode is not None:
            backend_kwargs["api_mode"] = args.openai_api_mode
        if args.openai_reasoning_effort is not None:
            backend_kwargs["reasoning_effort"] = args.openai_reasoning_effort
        if args.openai_max_output_tokens is not None:
            backend_kwargs["max_output_tokens"] = int(args.openai_max_output_tokens)
    elif args.backend.strip().lower().startswith("anthropic:"):
        if args.anthropic_temperature is not None:
            backend_kwargs["temperature"] = float(args.anthropic_temperature)
        if args.anthropic_max_tokens is not None:
            backend_kwargs["max_tokens"] = int(args.anthropic_max_tokens)
        if args.anthropic_thinking_budget_tokens is not None:
            backend_kwargs["thinking_budget_tokens"] = int(args.anthropic_thinking_budget_tokens)
    elif args.backend.strip().lower().startswith("gemini:"):
        if args.gemini_thinking_level is not None:
            backend_kwargs["thinking_level"] = str(args.gemini_thinking_level)
        if args.gemini_max_output_tokens is not None:
            backend_kwargs["max_output_tokens"] = int(args.gemini_max_output_tokens)
        if args.gemini_response_mime_type is not None:
            backend_kwargs["response_mime_type"] = str(args.gemini_response_mime_type)
        if args.gemini_timeout_seconds is not None:
            backend_kwargs["timeout_seconds"] = float(args.gemini_timeout_seconds)

    meta = run_stage7(
        project_root=Path(args.project_root),
        backend_spec=args.backend,
        information_condition=args.information_condition,
        conditions=conditions,
        modes=modes,
        reference_draw_seed=(args.reference_draw_seed if args.reference_draw_seed is not None else args.seed),
        backend_kwargs=backend_kwargs,
        max_concurrency=args.max_concurrency,
        resume=not args.no_resume,
        checkpoint_path=(Path(args.checkpoint_path) if args.checkpoint_path else None),
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        icc_probe=bool(args.icc_probe),
        firm_name_lookup_path=(Path(args.firm_name_lookup) if args.firm_name_lookup else None),
        icc_probe_tolerance=float(args.icc_probe_tolerance),
        candidate_library_quantile=int(args.candidate_library_quantile),
        row_id_file=(Path(args.row_id_file) if args.row_id_file else None),
        sample_size=args.sample_size,
        sample_seed=int(args.sample_seed),
        sample_strata=sample_strata,
        freeform_l1_budget=args.freeform_l1_budget,
        budgeted_conditions=budgeted_conditions,
        budget_contract_label=args.budget_contract_label,
        budget_tolerance=float(args.budget_tolerance),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
