# Thesis reproduction tools

This directory contains only the public stage runners required for a fresh
reproduction workspace. Python implementation belongs under `src/credit_recourse`;
PowerShell here only resolves the environment, fixes the experiment profile, and
invokes those modules.

## 1. Fresh workspace layout

Create an empty folder and place only the distributed `src/`, `tools/`, and raw
data below it. The raw inputs may be either the extracted canonical folders or
the three original archives (`raw_all.zip`, `raw_nonfinancial.zip`,
`rating_sample.zip`) placed in the project root or `data/raw_archives/`:

```text
thesis_repo_reproduction/
  src/
    credit_recourse/
  tools/
  raw_all.zip                 # accepted alternative to extracted data/raw/raw_all
  raw_nonfinancial.zip        # accepted alternative to extracted data/raw/raw_nonfinancial
  rating_sample.zip           # accepted alternative to extracted data/raw/rating_sample
  data/
    raw/                       # setup_env.ps1 creates this from the archives when absent
      raw_all/
      raw_nonfinancial/
        kospi_kosdaq/
        konex_optional/       # optional
      rating_sample/
```

Do not copy old `data/final_freeze`, `data/analysis`, checkpoints, or LLM run
archives into a clean reproduction.

## 2. Environment

```powershell
$Root = "C:\path\to\thesis_repo_reproduction"
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\setup_env.ps1" `
  -ProjectRoot $Root
```

`setup_env.ps1` first materializes the canonical raw directories from the three
archives when needed, installs the pinned environment (including `matplotlib`),
then validates and hashes the source, tools, and raw-input layout into:

```text
data/reproduction/workspace_manifest.json
```

## 3. Canonical stage order

### Oracle

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task Oracle
```

### RL

This first creates the LoopA/B2 substrate artifact, then runs the fixed final
Stage2-6 paper preset.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task RL
```

### Complete paper LLM profile

The LLM task runs and archives:

- GPT-5.4-mini IC-a/IC-b/IC-c full Stage7-9 grids;
- GPT-4.1-mini IC-b supplementary grid;
- Claude Haiku 4.5 IC-b supplementary grid;
- N5 L1=1.27 IC-a/IC-b/IC-c;
- same-date IC-b N5 generation-time budget frontier at 0.75/1.27/2.00/unbounded with within-arm C4 controls;
- optional `-MatchedBudgetC4` factorial mode applying the same finite-arm L1 contract to both C4 and C6;
- IC-c probe-only diagnostic.

All outputs are immutable archives below `data/final_freeze/llm_runs/<label>`.
The command requires explicit spend confirmation.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task LLM `
  -ConfirmLiveApiSpend -PromptApiKeys
```

### Existing-run IC-c probe only

When the archived IC-c Stage7-9 run predates the probe patch, run only the
575-row live probe through the same canonical entry point. This does not rerun
the 6,900-request Stage7 grid or Stage8/9. The default base and output labels are
`ICc_gpt54mini_p50_live_seed1` and
`ICc_probe_gpt54mini_p50_seed1_20260711`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task ICcProbe `
  -ConfirmLiveApiSpend -PromptApiKeys
```

The only new top-level archive directory is:

```text
data/final_freeze/llm_runs/ICc_probe_gpt54mini_p50_seed1_20260711/
```

Use `-ICcProbeBaseRunLabel` or `-ICcProbeRunLabel` only when the canonical
labels differ. Reusing the same output label resumes an incomplete checkpoint;
a completed immutable archive fails rather than creating a duplicate folder.

### One post-freeze analysis command

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_postfreeze_analysis.ps1" `
  -ProjectRoot $Root
```

All paper-used post-freeze analyses are generated under one tree:

```text
data/analysis/paper_repro/
  00_manifest/
  01_substrate_validation/
  02_llm_hypothesis_tests/
  03_output_contract_diagnostics/
  04_paper_assets/
  99_verification/
```

The analysis command never calls a live API. It fails on missing, ambiguous, or
contract-incompatible frozen inputs. The canonical run now also rebuilds thesis
Section 9.8 immediately after the N5M post-hoc frame is created:

- a pre-action, firm-grouped out-of-fold Ridge selector for the four N5M action
  budgets, run separately for C4 and C6; and
- a post-C4/pre-C6, firm-grouped out-of-fold Ridge gate using exactly
  `c4_final_l1`, `c4_projection_distance`, and `c4_active_dimensions`.

The producer persists the full firm-level OOF predictions, fold assignments,
feature order/hash, coefficients, input hashes, and summary arithmetic under:

```text
data/analysis/paper_repro/03_output_contract_diagnostics/n5m_adaptive_selection/
```

The frozen implementation contract is
`data/final_freeze/configs/n5m_adaptive_selection_contract.json`. Protected
Stage2 audit/label columns may remain in the upstream panel, but the selector
uses only the explicit current-state allow-list. Every firm's four budget rows
remain in the same held-out fold. The outputs are exploratory OOF diagnostics,
not independent-cohort validation. If the rebuilt values differ from an older
unarchived Section 9.8 calculation, update the thesis from the rebuilt outputs;
do not tune seeds/features to recover stale numbers.

Standalone producer and verifier commands for diagnosis are:

```powershell
python -m credit_recourse.analysis.n5m_adaptive_selection `
  --project-root $Root `
  --firm-frame "$Root\data\analysis\paper_repro\03_output_contract_diagnostics\n5m_posthoc\n5m_firm_frame.parquet" `
  --state-panel "$Root\data\final_freeze\stage2_candidate_projection\phase_eval_candidate.parquet" `
  --contract "$Root\data\final_freeze\configs\n5m_adaptive_selection_contract.json" `
  --output-dir "$Root\data\analysis\paper_repro\03_output_contract_diagnostics\n5m_adaptive_selection"

python -m credit_recourse.verification.verify_n5m_adaptive_selection_contract `
  --project-root $Root `
  --analysis-dir "$Root\data\analysis\paper_repro"
```

Contract-faithful synthetic regression:

```powershell
python -m credit_recourse.verification.verify_n5m_adaptive_selection_synthetic
```

To rebuild cleanly while preserving the old analysis as a dated backup:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_postfreeze_analysis.ps1" `
  -ProjectRoot $Root -ReplaceOutput
```

## 4. Public tools and responsibilities

| File | Responsibility |
|---|---|
| `setup_env.ps1` | Environment and fresh-workspace/raw provenance |
| `run_thesis_repro.ps1` | Canonical Oracle/RL/LLM/Analysis stage entry point |
| `run_oracle_stage0_stage1.ps1` | Oracle Stage0-1 implementation runner |
| `run_loopA_loopB2_stage2_extension.ps1` | Required substrate LoopA/B2 artifact |
| `run_rl_unified_stage3456.ps1` | Final RL Stage2-6 runner |
| `run_llm789_fresh_all_single_repo.ps1` | One homogeneous Stage7-9 LLM matrix |
| `run_n5_main_gpt54mini_icab.ps1` | N5 L1=1.27 IC-a/b/c matrix |
| `run_n5_budget_frontier_icb.ps1` | Same-batch IC-b generation-time budget frontier |
| `run_postfreeze_analysis.ps1` | Single all-paper post-freeze analysis runner |

`run_postfreeze_analysis.ps1` is the only public post-freeze analysis entry
point. Individual Python analysis modules remain internal, testable components.

If canonical analysis fails after the full N1 frontier has completed, recover
without rerunning the pre-frontier stages:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task AnalysisResume `
  -FailedAnalysisPath "$Root\data\archive\paper_repro_rebuilds\failed_<timestamp>"
```

Omit `-FailedAnalysisPath` to select the newest `failed_*` archive. The resume
path validates the completed frontier and runs only the remaining canonical
steps before promoting the archive back to `data\analysis\paper_repro`.

## 5. Analyses remaining after an existing `-Task Analysis` run

Do not interrupt or overlap an active canonical analysis process. After its
manifest status becomes `PASS`, execute the remaining no-API work through the
same canonical entry point:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task RemainingNoApi
```

This command does not perform final end-to-end reproduction and does not call an
LLM API. It:

- validates or repairs the three 1,000-draw row-shuffle cells for each primary
  information condition (unconditional, within-industry, within-rating-band);
- runs the frozen reference-quality/adoption response analysis; and
- seals the RL/LLM P50 candidate-library provenance report.

The optional 4,600-request IC-b C4/C6 generation-time frontier is explicit:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task BudgetFrontier `
  -ConfirmLiveApiSpend -PromptApiKeys
```

To run the matched-budget C4/C6 factorial frontier instead of the legacy C6-only budget frontier:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task BudgetFrontier `
  -MatchedBudgetC4 `
  -ConfirmLiveApiSpend -PromptApiKeys
```

This produces `N5M_*` archives under the separate run role
`paper_n5_matched_budget_frontier_icb`; it does not overwrite legacy `N5F_*` archives.
The matched analysis reports the C4 budget main effect, the fixed-budget
C6-minus-C4 reference/revision effect, and the interaction DID.

The Tier-A C4R experiment is integrated into the same existing frontier runner
and the same top-level entry point. It runs fresh same-batch C4/C4R/C6 cells at
L1<=0.75 and unbounded, then executes the C4R matched inference automatically:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task BudgetFrontier `
  -C4RMatched `
  -ConfirmLiveApiSpend -PromptApiKeys
```

A noncanonical N5M replication group also uses the same runner; no separate
replication wrapper is required:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task BudgetFrontier `
  -MatchedBudgetC4 `
  -ReplicationGroupId seed2 `
  -ExperimentSeed 2 -ExperimentReferenceDrawSeed 2 `
  -ConfirmLiveApiSpend -PromptApiKeys
```

The evaluator-only N5M OAT/Shapley analysis is integrated into the existing
post-freeze analysis runner. If exactly one canonical archive exists for each
required arm it is auto-resolved; otherwise pass explicit arm paths:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task Analysis `
  -RunN5MAxisSwap `
  -AxisSwapShapleyPermutations 64
```

There are no separate public C4R, N5M-replication, or axis-swap PowerShell
runners. Their implementation is consolidated into
`run_n5_budget_frontier_icb.ps1`, `run_postfreeze_analysis.ps1`, and
`run_thesis_repro.ps1`.

To execute both phases in sequence:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root -Task RemainingAll `
  -ConfirmLiveApiSpend -PromptApiKeys
```

Use `-PlanOnly` to inspect either command without writing analysis outputs or
sending API requests. In plan-only mode, do not pass `-ConfirmLiveApiSpend` or
`-PromptApiKeys`; the top-level runner must not require spend confirmation or
prompt for secrets when no live request can be issued.

## Repeated shuffle performance and resume contract

The 1,000-draw row-shuffle analyses use the same exact simulator and Oracle
substrate as the original ablation, but additional draws score only the target
`C6/free_form_10d` rows. The untouched Stage7 rows do not enter the permutation
statistic and must not be re-simulated for every seed.

Runtime behavior:

- automatic bounded parallelism: up to 8 worker processes;
- progress lines such as `[shuffle] completed=...`;
- checkpoint every 10 completed draws;
- resume from `shuffle_per_draw_summary.partial.csv` and
  `shuffle_permutation_checkpoint.json`;
- checkpoint input/signature mismatch is a hard failure;
- the first draw still writes the full auditable ablation artifacts.

The canonical entry point remains unchanged:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root `
  -Task AnalysisResume
```

Before applying a performance hotfix, stop any older running
`paper_repro_analysis` / `remaining_thesis_analyses` processes. Do not run two
analysis-resume jobs against the same output directory concurrently.

Worker override for Windows/local resource control:

```powershell
$env:CREDIT_RECOURSE_SHUFFLE_WORKERS = "4"
```

An explicit `--shuffle-workers` CLI value takes precedence; otherwise the environment value is used, then automatic selection (up to 8).

## Budget-frontier interrupted-run recovery

The frontier archive transaction uses a short `.txn-<uuid>` staging directory
so long frontier run labels and checkpoint filenames remain Windows-safe. If a
frontier arm finished Stage7–9 but failed during archiving, rerun the integrated
entry point with the original date tag so its request checkpoint is preserved
and resumed rather than discarded:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_thesis_repro.ps1" `
  -ProjectRoot $Root `
  -Task BudgetFrontier `
  -ConfirmLiveApiSpend `
  -PromptApiKeys `
  -BudgetFrontierDateTag "20260712_083213"
```

The runner preserves the matching checkpoint before canonical Stage7–9 cleanup,
restores it, and re-enters Stage7 with resume enabled. Completed request keys do
not trigger new API calls. Existing immutable arm archives are manifest-checked
and skipped. A directory without an archive manifest, an identity mismatch, or
checkpoint preservation failure remains a hard failure.

## Loop B2 / Stage1 B1 sign-alignment diagnostic

`run_loopB2_b1_b2_sign_diagnosis.ps1` is a fail-fast diagnostic-only runner for
`credit_recourse.oracle.verification.diagnose_loopb2_b1_alignment`. It never
changes the Loop B2 gate verdict. Use `-PlanOnly` to print the exact command
without reading inputs or writing outputs; use `-FailOnInversionSuspect` only
when an inversion diagnosis should be promoted to exit code 2.

## 6. Journal extension: amended GPT/Gemini four-arm C4/C4R/C6 grid

The frozen thesis E2 producer and the original v1 journal preregistration remain
unchanged. Haiku 4.5 remains a raw-budget feasibility failure. Gemini 3.5 Flash
with thinking-low/maxOutputTokens-1200 is also retained as a provider-feasibility
failure because MAX_TOKENS truncation stopped the first C4 arm before completion.
The v3 amendment replaces the replication cohort with stable Gemini 3.1
Flash-Lite, keeps thinking level low, raises maxOutputTokens to 4096, and requires
reuse of the four already completed GPT-5.4-mini archives.

The active machine-readable amendment is:

```text
src/credit_recourse/configs/c4r_journal_extension_prereg_v3.json
```

The amendment does not relax any prompt, parser, budget, row-alignment,
candidate-library, Stage7/8/9, or multiplicity contract.

### 6.1 Locate the previous failed-grid manifest

The failed live grid manifest is still useful because its four GPT arms have
`PASS` status. For the run started with date tag `20260717_145222`, the expected
path is:

```powershell
$Root = "C:\Users\Demian\Desktop\thesis_repo"
$OldGridManifest = Join-Path $Root `
  "data\analysis\c4r_journal_extension\live_20260717_145222\c4r_journal_grid_manifest.json"

Test-Path $OldGridManifest
```

The result must be `True`. The amended runner re-verifies every reused archive;
it does not trust a prior `PASS` string by itself.

### 6.2 No-API plan

After the Gemini amendment code has been publicly committed, use the new commit
URL as `-PreregEvidence`. Plan-only does not require `GEMINI_API_KEY`, but the
GPT reuse manifest is still required because the v3 contract explicitly forbids
new GPT calls.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_c4r_journal_grid.ps1" `
  -ProjectRoot $Root `
  -PreregContractPath "$Root\src\credit_recourse\configs\c4r_journal_extension_prereg_v2.json" `
  -ReuseGridManifest $OldGridManifest `
  -PlanOnly
```

A full no-API scripted rehearsal remains available. Rehearsal intentionally
generates both synthetic cohorts and therefore does not require a reuse
manifest:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_c4r_journal_grid.ps1" `
  -ProjectRoot $Root `
  -PreregContractPath "$Root\src\credit_recourse\configs\c4r_journal_extension_prereg_v2.json" `
  -ScriptedRehearsal
```

### 6.3 Recommended Gemini `0p75` QC pilot

Set only the Gemini key in the current PowerShell session. Reused GPT arms do
not require `OPENAI_API_KEY`.

```powershell
$GeminiKey = Read-Host "GEMINI_API_KEY" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $GeminiKey).Password

$AmendmentEvidence = "https://github.com/<owner>/<repo>/commit/<actual-amendment-commit>"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_c4r_journal_grid.ps1" `
  -ProjectRoot $Root `
  -PreregContractPath "$Root\src\credit_recourse\configs\c4r_journal_extension_prereg_v2.json" `
  -PreregEvidence $AmendmentEvidence `
  -CohortIds gemini31flashlite `
  -BudgetLabels 0p75 `
  -AllowIncompleteGrid
```

The pilot must finish with `PASS_PARTIAL_QC`. It uses the unchanged arm verifier
and therefore still requires 1,725 Stage7/8 rows, 1,150 Stage9 revision rows,
full prompt-payload hash integrity, and the preregistered finite-arm compliance
thresholds.

Record the pilot manifest path printed by the runner:

```powershell
$PilotGridManifest = "<printed PASS_PARTIAL_QC manifest path>"
```

### 6.4 Complete amended grid without repeating GPT or the pilot

After the Flash-Lite `0p75` pilot passes, reuse both the original GPT manifest and
the pilot manifest. Only Flash-Lite `1p27`, `2p00`, and `unbounded` are newly
called.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "$Root\tools\run_c4r_journal_grid.ps1" `
  -ProjectRoot $Root `
  -PreregContractPath "$Root\src\credit_recourse\configs\c4r_journal_extension_prereg_v2.json" `
  -PreregEvidence $AmendmentEvidence `
  -ReuseGridManifest $OldGridManifest,$PilotGridManifest
```

To skip the pilot and generate all four Gemini arms in one run, pass only
`$OldGridManifest`. The runner will reuse all four GPT arms and generate the
four Flash-Lite arms.

Axis attribution is evaluator-only and may be requested on the complete grid:

```powershell
  -RunAxisAttribution
```

It is intentionally forbidden on a partial pilot grid.

### 6.5 Amended-grid invariants

The runner hard-fails unless:

- the v3 amendment is locked and uses IC-b/P50/C4,C4R,C6;
- every reused GPT or Flash-Lite pilot archive independently passes arm QC;
- no `reuse_required` GPT arm is missing;
- Stage7/8/9 rows and firm keys align exactly;
- full prompt payload bytes are archived and hash-valid;
- finite-arm raw-budget compliance meets the original thresholds;
- the P50 candidate-library hash is identical across all eight analysis arms;
- Haiku and Gemini 3.5 Flash are excluded only from matched inference while their
  failed artifacts remain preserved as feasibility evidence; and
- GPT and Gemini 3.1 Flash-Lite are analyzed as separate cohorts, never pooled.

Primary combined outputs are produced by
`credit_recourse.analysis.c4r_matched_inference_v3`. The frozen thesis module
`c4r_matched_inference.py` and the v1 preregistration JSON are not modified or
replaced.
