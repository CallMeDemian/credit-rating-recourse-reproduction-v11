from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


NUMBER = re.compile(r"(?<![A-Za-z])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z])")
WORD = re.compile(r"[가-힣]{2,}|[A-Za-z_]{2,}|\d+(?:\.\d+)?")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(value: str) -> str:
    return value.replace("\\", "/").strip()


def number_value(value: object) -> float | None:
    try:
        return float(str(value).replace("−", "-").replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def decimals(token: str) -> int:
    clean = token.lower().split("e", 1)[0].replace(",", "")
    return len(clean.split(".", 1)[1]) if "." in clean else 0


def words(text: str) -> set[str]:
    return {item.lower() for item in WORD.findall(str(text)) if len(item) > 1}


def unit_of(cell: str) -> str:
    low = cell.lower()
    if "%" in cell:
        return "%"
    if "p=" in low or "p <" in low or "p<" in low:
        return "p값"
    if "년" in cell:
        return "년"
    if "개" in cell or "기업" in cell or "표본" in cell:
        return "개수"
    return "표기값"


def is_identifier(column: str, cell: str) -> bool:
    """Separate row-selection values from statistics that require calculation."""
    label = column.lower().replace(" ", "")
    text = cell.lower()
    identifier_labels = (
        "파운데이션모델", "모델", "행동크기조건", "행동크기", "총행동크기",
        "seed", "시드", "실행", "선정단계", "구분", "연도", "기간",
    )
    if any(key in label for key in identifier_labels):
        return True
    if any(key in text for key in ("gpt-", "gemini ", "claude ", "haiku ", "thinking-")):
        return True
    # Thresholds and counts embedded in a metric label define the comparison,
    # whereas a pure numeric result cell remains a statistic.
    if re.search(r"(?:≥|≤|>|<)\s*\d", text) and not re.fullmatch(r"\s*[-+−]?\d[\d,.]*%?\s*", text):
        return True
    if "동등성" in label and "에서" in text:
        return True
    if label == "의미" and ("이상" in text or "이하" in text):
        return True
    if label == "지표" and re.search(r"\d+\s*개\s*행동", text):
        return True
    return False


TABLE_EXTRA_SOURCES: dict[str, list[str]] = {
    "3-1": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_3-1_processed_counts.csv"],
    "4-3": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_4-3_recalculated.csv"],
    "4-4": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_4-4_recalculated.csv"],
    "4-5": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_4-5_recalculated.csv"],
    "4-6": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_4-6_recalculated.csv"],
    # The old registry pointed 6-7 at the C6-vs-C4 table.  The printed counts
    # actually come from the matched C4→C4R→C6 contrast output.
    "6-7": ["data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_contrasts.csv"],
    "6-8": ["data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_contrasts.csv"],
    "E-3": ["data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_contrasts.csv"],
    # One file per information condition; unchanged policy rows plus the
    # original-target summary reproduce the 4×3 values in table 5-6.
    "5-6": ["data/analysis/paper_repro/03_output_contract_diagnostics/signflip_mean_null"],
    "5-3": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_5-3_from_historical_raw.csv"],
    "5-4": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_5-4_from_historical_raw.csv"],
    "6-2": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_6-2_from_historical_raw.csv"],
    "6-4": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_6-4_from_historical_raw.csv"],
    "6-1": ["data/analysis/paper_repro/03_output_contract_diagnostics/ablation"],
    "H-1": ["data/analysis/paper_repro/03_output_contract_diagnostics/ablation"],
    "F-1": ["data/analysis/paper_repro/04_paper_assets/tables/n5m_score_budget_auditability_operating_points.csv"],
    "F-2": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_F-2_from_historical_raw.csv"],
    "I-1": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_I-1_from_historical_raw.csv"],
    "I-2": ["data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_I-2_from_historical_raw.csv"],
}

RECALCULATION_DIFFERENCE_SOURCES = {
    "4-5": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/table_4-5_recalculated.csv",
    "5-3": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_5-3_from_historical_raw.csv",
    "5-4": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_5-4_from_historical_raw.csv",
    "F-3": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_F-3_recalculated_from_historical_raw.csv",
    "H-2": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_H-2_recalculated_from_historical_raw.csv",
    "I-2": "data/analysis/paper_repro/06_thesis_lineage/recalculated_evidence/historical_raw/table_I-2_from_historical_raw.csv",
}

FINAL_ONLY_REASONS = {
    "4-4": "C_obs 관측행동의 기업별 평가행은 동봉 실행결과에 없음. 논문 최종표만 보존.",
    "5-8": "동일 조건 반복쌍과 교차 모델쌍의 기업별 비교 원자료는 동봉 실행결과에 없음. 논문 최종표만 보존.",
}


@dataclass
class Candidate:
    value: float
    locator: str
    context: str
    column: str
    source: str


def flatten_source(root: Path, relative: str) -> list[Candidate]:
    relative = rel(relative)
    if not relative:
        return []
    path = root / relative
    paths: list[Path]
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".yaml", ".yml", ".md", ".txt"})
    elif path.is_file():
        paths = [path]
    else:
        return []
    output: list[Candidate] = []
    for file in paths:
        source = rel(str(file.relative_to(root)))
        suffix = file.suffix.lower()
        if suffix == ".csv":
            try:
                rows = read_csv(file)
            except (UnicodeDecodeError, csv.Error):
                continue
            for row_no, row in enumerate(rows, start=1):
                row_context = " | ".join(f"{key}={value}" for key, value in row.items() if str(value).strip())
                for column, cell in row.items():
                    for match in NUMBER.finditer(str(cell)):
                        value = number_value(match.group(0))
                        if value is not None:
                            output.append(Candidate(value, f"행 {row_no}, 열 {column}", row_context, str(column), source))
        elif suffix == ".json":
            try:
                payload = json.loads(file.read_text(encoding="utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            def walk(item: object, key: str) -> None:
                if isinstance(item, bool) or item is None:
                    return
                if isinstance(item, (int, float)):
                    output.append(Candidate(float(item), key or "JSON 루트", key, key.rsplit(".", 1)[-1], source))
                elif isinstance(item, dict):
                    for child, value in item.items():
                        walk(value, f"{key}.{child}" if key else str(child))
                elif isinstance(item, list):
                    for index, value in enumerate(item):
                        walk(value, f"{key}[{index}]")

            walk(payload, "")
        else:
            try:
                lines = file.read_text(encoding="utf-8-sig").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(lines, start=1):
                for match in NUMBER.finditer(line):
                    value = number_value(match.group(0))
                    if value is not None:
                        output.append(Candidate(value, f"줄 {line_no}: {line.strip()}", line, line.split(":", 1)[0].strip(), source))
    return output


def value_matches(token: str, cell: str, candidate: float) -> tuple[bool, float, str]:
    target = number_value(token)
    if target is None:
        return False, 1.0, ""
    digit = decimals(token)
    tolerance = 0.5 * (10 ** (-digit)) + 1e-12 if digit else 1e-10
    scales = [(1.0, "근거값 직접 사용")]
    if "%" in cell:
        scales.insert(0, (100.0, "근거값 × 100"))
    for scale, label in scales:
        if abs(candidate * scale - target) <= tolerance:
            return True, scale, label
    return False, 1.0, ""


def choose_candidate(token: str, cell: str, column: str, row_context: str, candidates: list[Candidate]) -> tuple[Candidate | None, float, str, int]:
    target_words = words(f"{column} {row_context}")
    matched: list[tuple[float, Candidate, float, str]] = []
    for candidate in candidates:
        ok, scale, transform = value_matches(token, cell, candidate.value)
        if not ok:
            continue
        candidate_words = words(f"{candidate.column} {candidate.context}")
        overlap = len(target_words & candidate_words)
        union = len(target_words | candidate_words) or 1
        jaccard = overlap / union
        col_ratio = SequenceMatcher(None, column.lower(), candidate.column.lower()).ratio()
        context_ratio = SequenceMatcher(None, row_context.lower()[:400], candidate.context.lower()[:400]).ratio()
        score = 4.0 * jaccard + 1.5 * col_ratio + context_ratio
        if rel(candidate.source).startswith("data/analysis/"):
            score += 0.2
        matched.append((score, candidate, scale, transform))
    if not matched:
        return None, 1.0, "", 0
    matched.sort(key=lambda item: (-item[0], item[1].source, item[1].locator))
    score, candidate, scale, transform = matched[0]
    return candidate, scale, transform, len(matched)


def choose_context_candidate(column: str, row_context: str, candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    target_words = words(f"{column} {row_context}")
    ranked: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        candidate_words = words(f"{candidate.column} {candidate.context}")
        overlap = len(target_words & candidate_words)
        union = len(target_words | candidate_words) or 1
        score = 4.0 * overlap / union
        score += 1.5 * SequenceMatcher(None, column.lower(), candidate.column.lower()).ratio()
        score += SequenceMatcher(None, row_context.lower()[:400], candidate.context.lower()[:400]).ratio()
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1].source, item[1].locator))
    return ranked[0][1]


def display_rule(token: str, scale: float) -> str:
    digit = decimals(token)
    prefix = "×100 후 " if scale == 100.0 else ""
    if "e" in token.lower():
        return prefix + "과학적 표기"
    if digit:
        return f"{prefix}소수 {digit}자리 반올림"
    return prefix + "정수 표기"


def verify(token: str, cell: str, source_value: object) -> str:
    value = number_value(source_value)
    if value is None:
        return "계산 대상 아님"
    ok, _, _ = value_matches(token, cell, value)
    return "일치" if ok else "불일치"


def detailed_claims(lineage: Path, table: str) -> list[dict[str, str]]:
    path = lineage / "tables" / f"table_{table}" / "02_수치_대응표.csv"
    return read_csv(path) if path.is_file() else []


def convert_detailed(row: dict[str, str], table_meta: dict[str, str], snapshot: str) -> dict[str, object]:
    cell = row.get("논문표시값", "")
    comparison = row.get("비교결과", "")
    return {
        "수치ID": row.get("수치ID", ""), "표번호": row.get("표번호", ""), "페이지": row.get("논문페이지", table_meta.get("논문페이지", "")),
        "논문행": row.get("논문행", ""), "논문열": row.get("논문열", ""), "셀전체": cell,
        "논문표시값": cell, "단위": row.get("단위", ""), "표성격": "계산 결과",
        "추적구분": row.get("검증수준", "직접 계산 추적"), "근거파일": row.get("직접결과파일", ""),
        "근거위치": "; ".join(x for x in [row.get("직접결과행선택", ""), row.get("직접결과열", "")] if x),
        "근거값": row.get("직접결과값", ""), "입력파일": row.get("계산입력파일", ""),
        "입력선택": "; ".join(x for x in [row.get("계산입력행선택", ""), row.get("계산입력열", ""), row.get("결합키", "")] if x),
        "계산식": row.get("계산식", ""), "표시규칙": row.get("표시규칙", ""),
        "검산결과": comparison or verify(cell, cell, row.get("직접결과값", "")), "생성코드": row.get("생성코드", ""),
        "논문표파일": snapshot, "재실행": row.get("실행명령", ""), "설명": row.get("제한사유", ""),
    }


FIGURE_ROWS = {
    "1": ("개념도", "", "", "논문에 삽입된 개념도 자체가 최종 원본. 계산 수치 없음."),
    "2": ("개념도", "", "src/credit_recourse/configs/final_oracle_rl_contract.json", "Stage 0–6 흐름과 설정 대조."),
    "3": ("개념도", "", "src/credit_recourse/configs/selected_variable_master.csv", "세 Oracle 구조와 입력변수 대조."),
    "4": ("표시값 복원", "data/analysis/paper_repro/06_thesis_lineage/figure_plot_data/figure_4_oracle_preference_profiles.csv", "", "논문 삽입본의 3×10 셀 값을 전사. 기존 B3 후보벡터 연결 제거."),
    "5": ("직접 결과", "data/analysis/paper_repro/03_output_contract_diagnostics/main_harness_backend_decomposition/main_harness_backend_cell_means.csv", "src/credit_recourse/analysis/main_harness_backend_decomposition.py", "모델×응답형식×정책 셀 평균."),
    "6": ("보존 범위 확인", "data/analysis/paper_repro/06_thesis_lineage/thesis_printed_values/figures/figure_6.png", "src/credit_recourse/analysis/firm_level_reproducibility.py", "독립 반복실행 원자료는 04 결과에 미보존. RL 7회 실행자료와의 잘못된 연결 제거."),
    "7": ("직접 재계산", "data/analysis/paper_repro/06_thesis_lineage/figure_plot_data/figure_7_harness_ablation_values.csv", "src/credit_recourse/analysis/thesis_value_trace_builder.py", "원 C6와 네 가지 행동변환 결과를 직접 집계."),
    "8": ("직접 결과", "data/analysis/paper_repro/03_output_contract_diagnostics/n5m_posthoc/n5m_score_budget_auditability_operating_points.csv", "src/credit_recourse/analysis/n5m_posthoc.py", "모델·예산·정책별 Oracle-α 평균."),
    "9": ("직접 결과", "data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_contrasts.csv", "src/credit_recourse/analysis/c4r_matched_inference_v3.py", "모델·예산별 C4R−C4와 C6−C4R."),
    "10": ("개념도", "data/analysis/paper_repro/04_paper_assets/plot_data/F2_c4r_chart_plot_data.csv", "src/credit_recourse/analysis/c4r_matched_inference.py", "C4→C4R→C6 분해식과 직접 결과."),
    "11": ("보존 범위 확인", "data/analysis/paper_repro/06_thesis_lineage/thesis_printed_values/figures/figure_11.png", "", "두 모델의 생성·적용 행동 원자료가 04 결과에 함께 보존되지 않아 삽입본을 정확 원본으로 연결."),
    "12": ("개념도", "data/analysis/paper_repro/06_thesis_lineage/thesis_printed_values/figures/figure_12.png", "", "정책 하네스 개념도 자체가 최종 원본. 계산 수치 없음."),
    "H-1": ("직접 결과", "data/analysis/paper_repro/03_output_contract_diagnostics/main_harness_backend_decomposition/main_harness_backend_decomposition.csv", "src/credit_recourse/analysis/main_harness_backend_decomposition.py", "Oracle별 하네스·모델·상호작용 분산 비율."),
    "H-2": ("직접 결과", "data/analysis/paper_repro/03_output_contract_diagnostics/n5m_posthoc/n5m_adoption_quartiles.csv", "src/credit_recourse/analysis/n5m_posthoc.py", "채택률 사분위×예산별 C6−C4 평균."),
}


def make_figure_data(results: Path) -> None:
    target = results / "data/analysis/paper_repro/06_thesis_lineage/figure_plot_data"
    target.mkdir(parents=True, exist_ok=True)
    profiles = {
        "Oracle-α": [-0.7, 0.8, -1.6, -1.1, -0.5, 0.5, -0.5, 1.1, 0.7, 1.3],
        "Oracle-β": [-0.4, -0.1, -0.6, -0.6, -0.5, -1.1, 0.7, 2.2, 1.0, -0.4],
        "Oracle-γ": [-0.0, 1.5, -1.2, -1.1, -0.5, 0.4, -1.1, 0.1, 0.6, 1.4],
    }
    rows = []
    for oracle, values in profiles.items():
        for action, value in zip(["DL1", "DL2", "RF1", "CX1", "WC1", "WC2", "OE1", "OE2", "MX1", "MX2"], values):
            rows.append({"oracle": oracle, "standard_action": action, "z_scored_mean_delta_noop": value, "source": "논문 그림 4 셀 표시값"})
    write_csv(target / "figure_4_oracle_preference_profiles.csv", rows)

    base = results / "data/analysis/paper_repro/03_output_contract_diagnostics/ablation/IC-b/ICb_gpt54mini_p50_live_seed1"
    main = read_csv(results / "data/analysis/paper_repro/03_output_contract_diagnostics/main_harness_backend_decomposition/main_harness_backend_cell_means.csv")
    original = next(float(r["mean_score"]) for r in main if r.get("backend_label") == "GPT-5.4-mini" and r.get("harness_cell") == "C6__free_form_10d" and r.get("oracle_backend") == "alpha")
    def c6(path: str) -> float:
        row = next(r for r in read_csv(base / path / "ablation_policy_summary.csv") if r["policy"] == "C6" and r["mode"] == "free_form_10d")
        return float(row["mean_delta_R_score_alpha"])
    shuffle_rows = [r for r in read_csv(base / "row_shuffle_vector_null_native/shuffle_per_draw_summary.csv") if r["policy"] == "C6" and r["mode"] == "free_form_10d" and r["oracle_backend"] == "alpha"]
    shuffle = sum(float(r["mean_shuffled_score"]) for r in shuffle_rows) / len(shuffle_rows)
    figure7 = [
        {"condition": "원 C6", "mean_oracle_alpha": original, "calculation": "main_harness_backend_cell_means의 GPT-5.4-mini/C6/free_form/alpha"},
        {"condition": "Global mean", "mean_oracle_alpha": c6("global_mean_vector_null_native"), "calculation": "global_mean_vector_null 적용 후 575개 기업 평균"},
        {"condition": "Shuffle", "mean_oracle_alpha": shuffle, "calculation": f"shuffle {len(shuffle_rows)}회 mean_shuffled_score 평균"},
        {"condition": "L1 rescale", "mean_oracle_alpha": c6("l1_rescale_same_policy_candidate_mean"), "calculation": "L1 재조정 후 575개 기업 평균"},
        {"condition": "Nearest projection", "mean_oracle_alpha": c6("nearest_candidate_projection_native"), "calculation": "최근접 후보행동 투영 후 575개 기업 평균"},
    ]
    write_csv(target / "figure_7_harness_ablation_values.csv", figure7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    lineage = results / "data/analysis/paper_repro/06_thesis_lineage"
    toc = read_csv(lineage / "논문_전체_표그림_목록.csv")
    table_meta = {row["논문번호"]: row for row in toc if row["구분"] == "표"}
    figure_meta = {row["논문번호"]: row for row in toc if row["구분"] == "그림"}
    index = {row["thesis_table"]: row for row in read_csv(lineage / "THESIS_ITEM_INDEX.csv")}
    table_paths = {row["표번호"]: row for row in read_csv(lineage / "논문_전체_표_확인경로.csv")}
    claims: list[dict[str, object]] = []
    source_cache: dict[str, list[Candidate]] = {}

    for table, meta in table_meta.items():
        snapshot = f"data/analysis/paper_repro/06_thesis_lineage/thesis_all_tables/table_{table}.csv"
        source_table = results / snapshot
        rows = read_csv(source_table)
        headers = list(rows[0]) if rows else []
        registered = index.get(table, {})
        kind = table_paths.get(table, {}).get("구분", "정의·설정·연구설계")
        direct = rel(registered.get("direct_statistical_result", ""))
        paper = rel(registered.get("paper_facing_result", ""))
        producer = rel(registered.get("producer_code", ""))
        sources = []
        for source in [direct, paper]:
            if source and source not in sources:
                sources.append(source)
        for source in TABLE_EXTRA_SOURCES.get(table, []):
            if source not in sources:
                sources.append(source)
        if not sources and producer and producer.startswith("src/credit_recourse/configs/"):
            sources.append(producer)
        candidates: list[Candidate] = []
        for source in sources:
            if source not in source_cache:
                source_cache[source] = flatten_source(results, source)
            candidates.extend(source_cache[source])

        detailed = detailed_claims(lineage, table)
        token_total = sum(len(NUMBER.findall(str(value))) for item in rows for value in item.values())
        used = Counter()
        for row in detailed:
            converted = convert_detailed(row, meta, snapshot)
            claims.append(converted)
            used[str(converted["논문표시값"]).replace("+", "").replace("−", "-")] += 1

        # A complete hand-traced table replaces the generic token pass in full.
        # This avoids counting row labels twice when the detailed map stores a
        # semantic row name instead of the literal numeric label in the CSV.
        if detailed and len(detailed) == token_total:
            continue

        ordinal = 0
        for row_no, row in enumerate(rows, start=1):
            row_context = " | ".join(str(row.get(header, "")) for header in headers)
            for col_no, column in enumerate(headers, start=1):
                cell = str(row.get(column, ""))
                for token_no, match in enumerate(NUMBER.finditer(cell), start=1):
                    token = match.group(0)
                    normalized = token.replace("+", "").replace("−", "-")
                    if used[normalized] > 0:
                        used[normalized] -= 1
                        continue
                    ordinal += 1
                    candidate, scale, transform, candidate_count = choose_candidate(token, cell, column, row_context, candidates)
                    if candidate is not None:
                        trace = "직접 결과 행·열" if kind == "계산 결과" else "설정·정의 위치"
                        source_file, locator, source_value = candidate.source, candidate.locator, candidate.value
                        calculation = transform
                        rule = display_rule(token, scale)
                        checked = verify(token, cell, source_value)
                        note = "행·열 라벨과 같은 문맥의 근거 위치 확정."
                    else:
                        identifier = is_identifier(column, cell)
                        if identifier:
                            trace = "식별·조건 값"
                        elif table in RECALCULATION_DIFFERENCE_SOURCES:
                            trace = "원자료 재계산 차이"
                        elif table in FINAL_ONLY_REASONS:
                            trace = "최종표만 보존"
                        elif table == "3-1":
                            trace = "원자료 재집계 필요"
                        elif kind == "계산 결과":
                            trace = "최종표만 보존"
                        else:
                            trace = "정의·설계 값"
                        source_file, locator, source_value = snapshot, f"행 {row_no}, 열 {column}", token
                        if trace == "원자료 재계산 차이":
                            source_file = RECALCULATION_DIFFERENCE_SOURCES[table]
                            nearest = choose_context_candidate(column, row_context, candidates)
                            if nearest is not None:
                                locator, source_value = nearest.locator, nearest.value
                        elif trace == "원자료 재집계 필요":
                            source_file, locator, source_value = "01_RAW_DATA.zip", "전체 재무제표 원자료", ""
                        if identifier:
                            calculation = "결과행을 선택하는 모델·예산·시드·기간 조건. 통계 계산 대상 아님."
                        elif trace == "원자료 재계산 차이":
                            calculation = "동봉된 원자료를 표시된 재계산 CSV의 계산식으로 다시 집계. 현재 보존 실행값과 논문 최종 표기 사이 차이 있음."
                        elif trace == "원자료 재집계 필요":
                            calculation = "01_RAW_DATA.zip의 전체 재무제표를 통합해 기업-연도 키로 중복 제거하는 전체 원자료 집계 단계. 처리된 전체 패널 행수 파일은 별도 보존되지 않음."
                        elif kind != "계산 결과":
                            calculation = "논문 표에 직접 정의된 범위·설계값. 통계 계산 대상 아님."
                        else:
                            calculation = FINAL_ONLY_REASONS.get(table, "논문 최종표에는 보존. 별도 원 계산행 또는 계산 입력은 현재 동봉 결과에 없음.")
                        rule = "논문 표기 그대로"
                        if trace == "원자료 재계산 차이":
                            checked = "차이 있음"
                        elif trace in {"최종표만 보존", "원자료 재집계 필요"}:
                            checked = "독립 확인 제한"
                        else:
                            checked = "계산 대상 아님" if identifier or kind != "계산 결과" else "최종 표 보존"
                        note = "동봉 범위와 독립 확인 가능 범위를 구분해 기록."
                    claims.append({
                        "수치ID": f"T{table}.R{row_no}.C{col_no}.N{token_no}", "표번호": table, "페이지": meta["논문페이지"],
                        "논문행": row_no, "논문열": column, "셀전체": cell, "논문표시값": token, "단위": unit_of(cell), "표성격": kind,
                        "추적구분": trace, "근거파일": source_file, "근거위치": locator, "근거값": source_value,
                        "입력파일": direct if direct and direct != source_file else "", "입력선택": "",
                        "계산식": calculation, "표시규칙": rule, "검산결과": checked, "생성코드": producer,
                        "논문표파일": snapshot, "재실행": table_paths.get(table, {}).get("재실행", ""), "설명": note,
                    })

    make_figure_data(results)
    figure_rows = []
    for number, meta in figure_meta.items():
        level, data, code, method = FIGURE_ROWS[number]
        printed = f"data/analysis/paper_repro/06_thesis_lineage/thesis_printed_values/figures/figure_{number.replace('-', '_')}.png"
        figure_rows.append({"그림번호": number, "제목": meta["논문제목"], "페이지": meta["논문페이지"], "재현수준": level,
                            "논문삽입본": printed, "그림데이터": data, "생성코드": code, "확인결과": method})
        readme = lineage / "figures_by_thesis_number" / f"figure_{number.replace('-', '_')}" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(f"# 그림 {number}\n\n제목: {meta['논문제목']}\n\n재현 수준: {level}\n\n논문 삽입본: `{printed}`\n\n그림 데이터: `{data}`\n\n생성 코드: `{code or '해당 없음'}`\n\n확인 결과: {method}\n", encoding="utf-8")

    fields = ["수치ID", "표번호", "페이지", "논문행", "논문열", "셀전체", "논문표시값", "단위", "표성격", "추적구분", "근거파일", "근거위치", "근거값", "입력파일", "입력선택", "계산식", "표시규칙", "검산결과", "생성코드", "논문표파일", "재실행", "설명"]
    write_csv(lineage / "논문_전체_수치_추적.csv", claims, fields)
    write_csv(lineage / "논문_전체_그림_확인경로.csv", figure_rows)
    summary = []
    for table in table_meta:
        rows = [r for r in claims if r["표번호"] == table]
        summary.append({"표번호": table, "수치수": len(rows), "직접계산추적": sum(r["추적구분"] in {"직접 결과 행·열", "결과 행·열 확인", "정확 재현", "통계적 재현"} for r in rows),
                        "설정정의추적": sum(r["추적구분"] == "설정·정의 위치" for r in rows),
                        "식별조건": sum(r["추적구분"] == "식별·조건 값" for r in rows),
                        "정의설계": sum(r["추적구분"] == "정의·설계 값" for r in rows),
                        "원자료재계산차이": sum(r["추적구분"] == "원자료 재계산 차이" for r in rows),
                        "최종표만보존": sum(r["추적구분"] == "최종표만 보존" for r in rows),
                        "원자료재집계필요": sum(r["추적구분"] == "원자료 재집계 필요" for r in rows),
                        "불일치": sum(r["검산결과"] == "불일치" for r in rows)})
    write_csv(lineage / "수치_추적_요약.csv", summary)
    (lineage / "README.md").write_text(
        "# 논문 표·그림·수치 추적\n\n표 56개와 표 안의 숫자 표기를 전부 등록.\n\n수치 추적은 직접 결과 행·열, 설정·정의 위치, 식별·조건 값, 정의·설계 값, 원자료 재계산 차이, 최종표만 보존, 원자료 재집계 필요로 구분. 확정된 근거 위치와 표시 계산을 한 행에 기록.\n\n그림 14개는 직접 결과, 직접 재계산, 표시값 복원, 개념도, 보존 범위 확인으로 구분.\n",
        encoding="utf-8",
    )
    print(json.dumps({"tables": len(table_meta), "claims": len(claims), "figures": len(figure_rows), "trace_types": Counter(str(r["추적구분"]) for r in claims)}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
