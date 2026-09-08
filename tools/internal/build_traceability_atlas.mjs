import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const outputPath = process.argv[2];
const qaDir = process.argv[3] || path.join(path.dirname(outputPath || "."), "_atlas_qa");
if (!outputPath) throw new Error("Usage: node build_traceability_atlas.mjs <output.pptx> [qa-dir]");

const W = 1280;
const H = 720;
const C = {
  bg: "#F8FAFC", navy: "#17324D", teal: "#0F766E", blue: "#2563EB",
  amber: "#D97706", red: "#B91C1C", slate: "#475569", gray: "#64748B",
  border: "#CBD5E1", white: "#FFFFFF", paleBlue: "#EAF2F8",
  paleTeal: "#E7F6F3", paleAmber: "#FEF3C7", paleRed: "#FEE2E2",
};
const FONT = "Malgun Gothic";

const deck = Presentation.create({ slideSize: { width: W, height: H } });

function box(slide, name, x, y, w, h, fill = C.white, line = C.border, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect", name,
    position: { left: x, top: y, width: w, height: h },
    fill, line: { style: "solid", fill: line, width: 1.2 }, borderRadius: radius,
  });
}

function textBox(slide, name, value, x, y, w, h, size = 16, color = C.slate, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { fontFamily: FONT, fontSize: size, color, bold, alignment: align, verticalAlignment: "middle" };
  return shape;
}

function header(slide, no, title, takeaway) {
  slide.background.fill = C.bg;
  slide.shapes.add({ geometry: "rect", name: `top-rule-${no}`, position: { left: 0, top: 0, width: W, height: 10 }, fill: C.teal, line: { style: "solid", fill: C.teal, width: 0 } });
  textBox(slide, `title-${no}`, `${no}. ${title}`, 44, 24, 1192, 48, 38, C.navy, true);
  textBox(slide, `takeaway-${no}`, takeaway, 46, 77, 1188, 34, 18, C.slate);
  textBox(slide, `footer-${no}`, "THESIS REPRODUCTION · TRACEABILITY ATLAS", 44, 688, 520, 18, 11, C.gray);
  textBox(slide, `page-${no}`, `${no} / 7`, 1180, 688, 56, 18, 11, C.gray, false, "right");
}

function addNode(slide, cfg) {
  const { id, x, y, w, h, title, pathText, producer, grain, meaning, fill = C.white, accent = C.teal, titleSize = 19, bodySize = 16 } = cfg;
  const card = box(slide, `node-${id}`, x, y, w, h, fill, C.border);
  slide.shapes.add({ geometry: "rect", name: `rail-${id}`, position: { left: x, top: y, width: 8, height: h }, fill: accent, line: { style: "solid", fill: accent, width: 0 } });
  textBox(slide, `node-title-${id}`, title, x + 18, y + 10, w - 30, 28, titleSize, C.navy, true);
  const lines = [];
  if (pathText) lines.push(`PATH  ${pathText}`);
  if (producer) lines.push(`PRODUCER  ${producer}`);
  if (grain) lines.push(`GRAIN  ${grain}`);
  if (meaning) lines.push(`MEANING  ${meaning}`);
  textBox(slide, `node-body-${id}`, lines.join("\n"), x + 18, y + 40, w - 30, h - 48, bodySize, C.slate);
  return card;
}

let edgeLabelCounter = 0;
function link(slide, from, to, label, opts = {}) {
  const color = opts.color || C.teal;
  const conn = slide.shapes.connect(from, to, {
    kind: opts.kind || "straight",
    fromSide: opts.fromSide || "right",
    toSide: opts.toSide || "left",
    line: { style: opts.dashed ? "dashed" : "solid", fill: color, width: 2.4 },
    head: { type: "arrow", width: "med", length: "med" },
  });
  const lx = opts.labelX ?? ((from.position.left + from.position.width + to.position.left) / 2 - 46);
  const ly = opts.labelY ?? ((from.position.top + from.position.height / 2 + to.position.top + to.position.height / 2) / 2 - 13);
  edgeLabelCounter += 1;
  const labelBg = box(slide, `edge-label-bg-${edgeLabelCounter}`, lx, ly, 92, 26, C.bg, C.bg, "rounded-sm");
  labelBg.bringToFront();
  const labelText = textBox(slide, `edge-label-${edgeLabelCounter}`, label, lx, ly + 2, 92, 22, 13, color, true, "center");
  labelText.bringToFront();
  return conn;
}

function callout(slide, name, value, x, y, w, h, fill, line, size = 16) {
  box(slide, `callout-${name}`, x, y, w, h, fill, line);
  textBox(slide, `callout-text-${name}`, value, x + 14, y + 7, w - 28, h - 14, size, C.slate);
}

function notes(slide, sources) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

{
  const s = deck.slides.add();
  header(s, 1, "저장소 구조와 5개 실행 모드", "심사위원이 만지는 공개 진입점은 2개뿐이며, 보존 경로와 clean 계산 경로를 섞지 않는다.");
  const dirs = [
    ["src", "원본 Stage0–9 + V10 분석", C.paleBlue], ["tools", "공개 runner 2개", C.paleTeal],
    ["contracts", "입력·출력 정의", C.white], ["analysis", "표·claim 계산 규칙", C.white],
    ["data", "현재 실행 결과", C.paleBlue], ["frozen_outputs", "불변 역사 snapshot", C.paleAmber],
    ["docs", "논문·계통도·실행기록", C.white],
  ];
  dirs.forEach(([name, role, fill], i) => {
    const x = 44 + i * 170;
    box(s, `dir-${name}`, x, 132, 154, 78, fill, C.border);
    textBox(s, `dir-name-${name}`, name, x + 12, 142, 130, 24, 17, C.navy, true, "center");
    textBox(s, `dir-role-${name}`, role, x + 8, 169, 138, 32, 13, C.slate, false, "center");
  });
  textBox(s, "runner-one", "tools/RUN_REPRODUCTION.ps1", 52, 230, 560, 30, 20, C.navy, true);
  const modes = [
    ["FrozenReplay", "frozen_outputs → 실제 V10 downstream 재계산", "학습·API 없음", C.teal],
    ["OracleClean", "raw → Stage0/1 → Oracle", "Stage0–1", C.blue],
    ["OracleRLClean", "raw → Oracle → RL", "Stage2–6", C.blue],
    ["OracleRLLLMClean", "raw → Oracle → RL → live LLM", "Stage7–9", C.amber],
    ["FullClean", "raw → Oracle → RL → LLM → analysis", "전체", C.amber],
  ];
  modes.forEach(([mode, flow, stop, color], i) => {
    const y = 270 + i * 64;
    box(s, `mode-${i}`, 52, y, 650, 50, C.white, C.border);
    box(s, `mode-no-${i}`, 62, y + 7, 42, 36, color, color, "rounded-md");
    textBox(s, `mode-no-text-${i}`, `${i + 1}`, 62, y + 7, 42, 36, 16, C.white, true, "center");
    textBox(s, `mode-name-${i}`, mode, 120, y + 7, 185, 36, 16, C.navy, true);
    textBox(s, `mode-flow-${i}`, flow, 300, y + 7, 310, 36, 15, C.slate);
    textBox(s, `mode-stop-${i}`, stop, 602, y + 7, 86, 36, 14, color, true, "right");
  });
  const run = addNode(s, { id: "run", x: 758, y: 257, w: 464, h: 140, title: "선택한 run", pathText: "data/runs/<run_id>", producer: "RUN_REPRODUCTION.ps1", grain: "실행 ID + stage별 실제 산출물", meaning: "현재 실행이 재현한 결과", fill: C.paleBlue });
  const excel = addNode(s, { id: "excel", x: 758, y: 446, w: 464, h: 140, title: "논문 표·그림 Excel", pathText: "data/thesis_outputs", producer: "BUILD_THESIS_OUTPUTS.ps1", grain: "논문 표·그림·핵심 본문수치", meaning: "선택 run의 수치를 다시 계산", fill: C.paleTeal });
  link(s, run, excel, "EXPORT", { fromSide: "bottom", toSide: "top", labelX: 944, labelY: 408 });
  callout(s, "frozen-parent", "frozen_outputs는 FrozenReplay의 역사 입력일 뿐이다. Clean mode의 계산 parent는 data/raw이며 frozen_outputs를 읽지 않는다.", 758, 606, 464, 58, C.paleAmber, C.amber, 15);
  notes(s, ["tools/RUN_REPRODUCTION.ps1", "tools/BUILD_THESIS_OUTPUTS.ps1", "README.md"]);
}

{
  const s = deck.slides.add();
  header(s, 2, "Raw → Stage0/1 → Oracle", "등급 firm-year panel과 재무 statement-item panel은 Stage0에서 따로 보존하고, Stage1에서 처음 결합한다.");
  const rawR = addNode(s, { id: "raw-rating", x: 44, y: 145, w: 265, h: 138, title: "등급 원자료", pathText: "data/raw/rating_sample/*.xlsx", producer: "Stage0 raw reader", grain: "firm-year", meaning: "평가사·등급 패널", fill: C.paleBlue });
  const st0R = addNode(s, { id: "st0-rating", x: 355, y: 145, w: 300, h: 138, title: "Stage0 등급 panel", pathText: "stage0/.../stage0_canonical_panel.parquet", producer: "build_stage0_foundation_from_raw.py", grain: "firm-year", meaning: "재무 panel과 별도 보존", bodySize: 15 });
  const rawF = addNode(s, { id: "raw-fin", x: 44, y: 350, w: 265, h: 138, title: "재무 원자료", pathText: "data/raw/raw_all/*.xlsx", producer: "Stage0 raw reader", grain: "firm-year-item", meaning: "재무제표 long panel", fill: C.paleBlue });
  const st0F = addNode(s, { id: "st0-fin", x: 355, y: 350, w: 300, h: 138, title: "Stage0 재무 panel", pathText: "stage0/.../statement_items_panel.parquet", producer: "build_stage0_foundation_from_raw.py", grain: "firm-year-item", meaning: "등급 panel과 별도 보존", bodySize: 15 });
  const join = addNode(s, { id: "stage1-join", x: 710, y: 220, w: 278, h: 176, title: "Stage1 결합·변수", pathText: "stage1_oracle_inputs/...", producer: "final_stage0_adapter.py", grain: "firm-year", meaning: "JOIN 후 비율계산·변수선정", fill: C.paleTeal });
  const oracle = addNode(s, { id: "oracle", x: 1034, y: 220, w: 202, h: 176, title: "Oracle α·β·γ", pathText: "stage1_oracle_backends/*", producer: "backends/*/pipeline.py", grain: "firm-year", meaning: "점수·등급 정렬 검증", fill: C.paleAmber, titleSize: 18, bodySize: 15 });
  link(s, rawR, st0R, "FILTER", { labelX: 300, labelY: 200 });
  link(s, rawF, st0F, "READ", { labelX: 300, labelY: 405 });
  link(s, st0R, join, "JOIN", { labelX: 635, labelY: 236 });
  link(s, st0F, join, "JOIN", { labelX: 635, labelY: 410 });
  link(s, join, oracle, "TRAIN / SCORE", { labelX: 970, labelY: 292 });
  callout(s, "finance-gap", "원본 producer에 별도 금융업 업종코드 제외가 확인되지 않았다. 결과를 바꾸는 새 필터는 넣지 않았으며, 이 차이는 논문 표본 claim의 남은 gap이다.", 55, 548, 1170, 70, C.paleRed, C.red, 16);
  callout(s, "frozen-counts", "보존 snapshot 예: 등급 firm-year 4,924행 · 재무 statement-item 77,077,533행 · Oracle-α 모델링 3,822행. 최신 값은 thesis Excel에서 source 행으로 다시 계산한다.", 55, 628, 1170, 44, C.paleAmber, C.amber, 14);
  notes(s, ["src/credit_recourse/oracle/stage0/build_stage0_foundation_from_raw.py", "src/credit_recourse/oracle/stage1/stage00_01_rating_statement/final_stage0_adapter.py", "data/final_freeze/stage0_oracle_foundation", "docs/KNOWN_LIMITATIONS.md"]);
}

{
  const s = deck.slides.add();
  header(s, 3, "Oracle·후보행동 → RL Stage2–6", "P50 후보와 counterfactual transition으로 Candidate-IQL을 학습하고, Stage6에서 같은 기업의 no-op과 비교한다.");
  const n1 = addNode(s, { id: "rl-input", x: 48, y: 150, w: 330, h: 142, title: "Oracle 상태 + 후보행동", pathText: "stage1 outputs + candidate library", producer: "Stage1 Oracle / config", grain: "firm-year × candidate", meaning: "상태·후보; 학습 reward에 Oracle 점수 없음", fill: C.paleBlue, bodySize: 15 });
  const n2 = addNode(s, { id: "stage2", x: 474, y: 150, w: 330, h: 142, title: "Stage2 후보·전이", pathText: "stage2_candidate_projection", producer: "final_stage2_candidate_projection", grain: "firm-year-candidate", meaning: "P50 후보행동 · SIMULATE", fill: C.paleTeal, bodySize: 15 });
  const n3 = addNode(s, { id: "stage3", x: 900, y: 150, w: 330, h: 142, title: "Stage3 encoder", pathText: "stage3_acd_ssl/*.pt", producer: "final_stage3_acd_ssl", grain: "firm-year representation", meaning: "self-supervised encoder", bodySize: 15 });
  const n4 = addNode(s, { id: "stage4", x: 900, y: 380, w: 330, h: 142, title: "Stage4 behavior cloning", pathText: "stage4_candidate_bc/*.pt", producer: "final_stage4_candidate_bc", grain: "firm-year → candidate", meaning: "행동 vocabulary 학습", bodySize: 15 });
  const n5 = addNode(s, { id: "stage5", x: 474, y: 380, w: 330, h: 142, title: "Stage5 Candidate-IQL", pathText: "stage5_candidate_iql/*.pt", producer: "final_stage5_candidate_iql", grain: "firm-year → candidate", meaning: "offline RL; Oracle score 미사용", fill: C.paleTeal, bodySize: 15 });
  const n6 = addNode(s, { id: "stage6", x: 48, y: 380, w: 330, h: 142, title: "Stage6 정책 평가", pathText: "stage6_*_eval", producer: "final_stage6_*_eval", grain: "firm × policy × oracle", meaning: "action score − same-firm no-op", fill: C.paleAmber, bodySize: 15 });
  link(s, n1, n2, "READ / SIMULATE", { labelX: 370, labelY: 205 });
  link(s, n2, n3, "TRAIN", { labelX: 804, labelY: 205 });
  link(s, n3, n4, "TRAIN", { fromSide: "bottom", toSide: "top", labelX: 1018, labelY: 320 });
  link(s, n4, n5, "TRAIN", { fromSide: "left", toSide: "right", labelX: 804, labelY: 435 });
  link(s, n5, n6, "SCORE", { fromSide: "left", toSide: "right", labelX: 370, labelY: 435 });
  callout(s, "rl-def", "정책가치 = 같은 row_id 기업의 행동 후 Oracle score − 같은 row_id 기업의 no-op Oracle score. 표·그림의 firm 평균은 이 paired 차이를 집계한다.", 74, 562, 1132, 74, C.paleTeal, C.teal, 17);
  notes(s, ["src/credit_recourse/rl/pipelines/final_stage2_candidate_projection/pipeline.py", "src/credit_recourse/rl/pipelines/final_stage5_candidate_iql/pipeline.py", "src/credit_recourse/rl/pipelines/final_stage6_candidate_iql_multi_oracle_eval/pipeline.py", "src/credit_recourse/configs/final_oracle_rl_contract.json"]);
}

{
  const s = deck.slides.add();
  header(s, 4, "LLM Stage7 → projection → Stage8 → Stage9", "Stage7 원응답과 제안행동을 보존하고, 적용행동을 별도로 만든 뒤 C4에서 C4R과 C6이 각각 분기한다.");
  const ref = addNode(s, { id: "llm-ref", x: 42, y: 178, w: 222, h: 158, title: "Stage6 reference", pathText: "stage6_candidate_selector_eval", producer: "Candidate-IQL", grain: "firm-year", meaning: "RL 참조후보 + no-op", fill: C.paleBlue, bodySize: 15 });
  const gen = addNode(s, { id: "stage7-gen", x: 314, y: 138, w: 286, h: 172, title: "Stage7 원응답", pathText: "llm_runs/<run>/stage7_*", producer: "Stage7 generation", grain: "firm × condition × mode", meaning: "model·prompt·params·raw response", bodySize: 15 });
  const apply = addNode(s, { id: "stage7-apply", x: 314, y: 360, w: 286, h: 172, title: "parse · projection · C4", pathText: "initial_actions_by_key", producer: "response_parser + projector", grain: "firm × mode", meaning: "제안행동 ≠ 적용행동; L1 지시 ≠ clipping", fill: C.paleTeal, bodySize: 15 });
  const c4r = addNode(s, { id: "c4r", x: 660, y: 140, w: 250, h: 158, title: "C4R 자기재검토", pathText: "C4 → C4R", producer: "2차 prompt", grain: "firm × mode", meaning: "C4를 읽고 스스로 수정", fill: C.paleAmber, bodySize: 15 });
  const c6 = addNode(s, { id: "c6", x: 660, y: 370, w: 250, h: 158, title: "C6 외부참조", pathText: "C4 → C6/C6X", producer: "2차 prompt", grain: "firm × mode", meaning: "C4 + 외부 정책 참조", fill: C.paleAmber, bodySize: 15 });
  const eval8 = addNode(s, { id: "stage8", x: 970, y: 154, w: 266, h: 172, title: "Stage8 Simulator·Oracle", pathText: "stage8_llm_multi_oracle_eval", producer: "final_stage8_*", grain: "firm × policy × oracle", meaning: "SIMULATE → SCORE → same-firm delta", fill: C.paleBlue, bodySize: 15 });
  const comp9 = addNode(s, { id: "stage9", x: 970, y: 384, w: 266, h: 144, title: "Stage9 비교", pathText: "stage9_llm_rl_comparison", producer: "final_stage9_*", grain: "firm × contrast × oracle", meaning: "LLM vs RL paired 비교", fill: C.paleTeal, bodySize: 15 });
  link(s, ref, gen, "READ", { labelX: 252, labelY: 220 });
  link(s, gen, apply, "PARSE / PROJECT", { fromSide: "bottom", toSide: "top", labelX: 410, labelY: 320 });
  link(s, apply, c4r, "BRANCH", { labelX: 592, labelY: 274 });
  link(s, apply, c6, "BRANCH", { labelX: 592, labelY: 444 });
  link(s, c4r, eval8, "SIMULATE / SCORE", { labelX: 900, labelY: 216 });
  link(s, c6, eval8, "SIMULATE / SCORE", { labelX: 900, labelY: 350 });
  link(s, eval8, comp9, "AGGREGATE", { fromSide: "bottom", toSide: "top", labelX: 1058, labelY: 344 });
  callout(s, "branch-truth", "C4R과 C6은 모두 C4에서 시작하는 형제 조건이다. C6−C4R은 결과 contrast일 뿐 C4→C4R→C6 순차 처리라는 뜻이 아니다.", 56, 572, 1168, 72, C.paleRed, C.red, 17);
  notes(s, ["src/credit_recourse/rl/pipelines/final_stage7_llm_action_generation/pipeline.py", "src/credit_recourse/rl/pipelines/final_stage7_llm_action_generation/response_parser.py", "src/credit_recourse/rl/pipelines/final_stage8_llm_multi_oracle_eval/pipeline.py", "src/credit_recourse/rl/pipelines/final_stage9_llm_rl_comparison/pipeline.py"]);
}

{
  const s = deck.slides.add();
  header(s, 5, "실행 결과 → 분석 → 논문 표·그림·본문 수치", "Excel은 논문 인쇄값을 복사하지 않고, 선택 run의 원행과 producer 집계를 source cell부터 다시 연결한다.");
  const current = addNode(s, { id: "current-output", x: 40, y: 154, w: 270, h: 152, title: "선택 run 산출물", pathText: "data/final_freeze + llm_runs", producer: "원본 Stage0–9", grain: "firm/action/policy/oracle", meaning: "Oracle·RL·LLM 실제 결과", fill: C.paleBlue, bodySize: 15 });
  const v10 = addNode(s, { id: "v10-analysis", x: 355, y: 154, w: 270, h: 152, title: "V10 확장 분석", pathText: "src/credit_recourse/analysis", producer: "paper_repro + E2/E3/E4", grain: "contrast/model/budget/firm", meaning: "FILTER · JOIN · paired test", fill: C.paleTeal, bodySize: 15 });
  const analysis = addNode(s, { id: "analysis-output", x: 670, y: 154, w: 270, h: 152, title: "분석 산출물", pathText: "data/analysis/paper_repro", producer: "V10 analysis modules", grain: "논문 contrast/표 행", meaning: "CSV·Parquet·JSON 계산결과", bodySize: 15 });
  const excel = addNode(s, { id: "thesis-excel", x: 985, y: 154, w: 250, h: 152, title: "논문 Excel", pathText: "data/thesis_outputs", producer: "BUILD_THESIS_OUTPUTS.ps1", grain: "표·그림·핵심 본문수치", meaning: "편집 가능한 workbook", fill: C.paleAmber, bodySize: 15 });
  link(s, current, v10, "READ", { labelX: 298, labelY: 214 });
  link(s, v10, analysis, "AGGREGATE", { labelX: 612, labelY: 214 });
  link(s, analysis, excel, "EXPORT", { labelX: 926, labelY: 214 });
  const docx = addNode(s, { id: "docx", x: 74, y: 382, w: 330, h: 150, title: "논문 DOCX inventory", pathText: "docs/thesis/canonical_thesis.docx", producer: "OOXML inventory reader", grain: "57개 표 + 14개 그림 + 본문 claim", meaning: "번호·제목·위치를 동적으로 읽음", bodySize: 15 });
  const calc = addNode(s, { id: "excel-calc", x: 472, y: 364, w: 402, h: 186, title: "Excel 내부 계산선", pathText: "SOURCE → CALCULATION → PAPER_TABLE / CHART_DATA", producer: "prepare.py + workbook builder", grain: "선택된 source 행·cell", meaning: "계산식·source·producer·run_id를 함께 표시", fill: C.paleTeal, bodySize: 15 });
  const paper = addNode(s, { id: "paper", x: 942, y: 382, w: 294, h: 150, title: "심사위원이 확인", pathText: "표·그림·NUMERIC_CLAIMS", producer: "Excel 수식/원행 추적", grain: "논문 item", meaning: "실행 결과에서 계산됐는지 확인", bodySize: 15 });
  link(s, docx, calc, "INVENTORY", { labelX: 390, labelY: 442 });
  link(s, calc, paper, "READ / EDIT", { labelX: 860, labelY: 442 });
  callout(s, "no-copy", "계산 입력으로 쓰지 않음: 논문 인쇄값 · V10 TABLE_VALUES · 기존 paper CSV/PNG. 이들은 비교 reference일 뿐 compute parent가 아니다.", 72, 586, 1164, 58, C.paleRed, C.red, 16);
  notes(s, ["src/credit_recourse/reproduction/thesis_outputs/prepare.py", "src/credit_recourse/reproduction/thesis_outputs/build_workbooks.py", "analysis/thesis_output_mappings/thesis_output_plan.json", "tools/BUILD_THESIS_OUTPUTS.ps1"]);
}

{
  const s = deck.slides.add();
  header(s, 6, "장·부록별 논문 terminal", "DOCX에서 동적으로 읽은 71개 표·그림을 실제 producer 영역과 연결한다. 비경험적 설계표는 계산값처럼 취급하지 않는다.");
  const cols = [44, 250, 455, 800];
  const widths = [206, 205, 345, 436];
  ["논문 구간", "terminal", "실제 producer 영역", "심사위원이 확인할 핵심"].forEach((value, i) => {
    box(s, `th-${i}`, cols[i], 138, widths[i], 46, C.navy, C.navy, "rounded-sm");
    textBox(s, `th-text-${i}`, value, cols[i] + 6, 143, widths[i] - 12, 36, 16, C.white, true, "center");
  });
  const rows = [
    ["3장 데이터·설계", "T3-1…T3-6", "Stage0/1 · candidate · 조건 정의", "표본 흐름·설계; 경험적 셀만 계산"],
    ["4장 Oracle·RL", "T4-1…T4-6 · F4", "Oracle validation · Stage2 · Stage6", "Oracle 타당성·Simulator·Candidate-IQL"],
    ["5장 LLM 본실험", "T5-1…T5-8 · F5…F8", "Stage7–9 · V10 post-freeze", "모델·조건·ablation·budget"],
    ["6장 확장분석", "T6-1…T6-9 · F9…F11", "C4R/N5M/E2–E4", "자기재검토·외부참조·축 기여"],
    ["부록 B–I", "TB…TI · FH", "동일 producer의 상세 산출물", "검정·민감도·반복성·추가 그림"],
    ["부록 J", "TJ-U", "구현 경로 설명", "경험적 계산 대상 아님"],
  ];
  rows.forEach((row, r) => {
    const y = 188 + r * 68;
    row.forEach((value, i) => {
      box(s, `tr-${r}-${i}`, cols[i], y, widths[i], 64, r % 2 ? C.white : C.paleBlue, "#E2E8F0", "rounded-sm");
      textBox(s, `tr-text-${r}-${i}`, value, cols[i] + 10, y + 8, widths[i] - 20, 48, 15, i === 0 ? C.navy : C.slate, i === 0);
    });
  });
  callout(s, "terminals", "동적 inventory 결과: 번호 있는 표 56 + 번호 없는 표 1 + 그림 14 = terminal 71개. 각 경험적 item의 최종 Excel은 선택 run에서 source 행이 실제로 해석될 때만 만든다.", 60, 610, 1160, 54, C.paleTeal, C.teal, 15);
  notes(s, ["docs/thesis/canonical_thesis.docx", "analysis/thesis_output_mappings/thesis_output_plan.json", "analysis/thesis_output_mappings/thesis_item_sources.json"]);
}

{
  const s = deck.slides.add();
  header(s, 7, "FrozenReplay와 Clean Run의 provenance·gap", "두 경로는 같은 downstream 분석을 쓰지만, 시작 입력과 재생성 범위가 다르므로 결과의 의미도 구분한다.");
  textBox(s, "frozen-heading", "FROZEN REPLAY", 62, 128, 500, 32, 22, C.teal, true);
  textBox(s, "clean-heading", "CLEAN RUN", 714, 128, 500, 32, 22, C.blue, true);
  const left = [
    ["불변 역사 입력", "frozen_outputs/final_freeze", "보존된 Oracle·RL·LLM"],
    ["작업 복사", "data/final_freeze", "보존본 자체는 수정하지 않음"],
    ["downstream 실제 재계산", "data/analysis", "V10 post-freeze · E2–E4"],
    ["논문 Excel", "data/thesis_outputs", "선택 run의 실제 수치"],
  ];
  const right = [
    ["원자료", "data/raw", "clean compute parent"],
    ["Stage0–6", "data/final_freeze", "Oracle·RL 실제 재학습"],
    ["Stage7–9", "llm_runs/<fresh run>", "live API · raw response 보존"],
    ["분석·논문 Excel", "data/analysis → thesis_outputs", "현재 실행의 결과"],
  ];
  const lnodes = [];
  const rnodes = [];
  left.forEach(([title, p, meaning], i) => lnodes.push(addNode(s, { id: `f-${i}`, x: 62, y: 170 + i * 103, w: 504, h: 82, title, pathText: p, meaning, fill: i === 0 ? C.paleAmber : C.white, accent: C.teal, titleSize: 17, bodySize: 14 })));
  right.forEach(([title, p, meaning], i) => rnodes.push(addNode(s, { id: `c-${i}`, x: 714, y: 170 + i * 103, w: 504, h: 82, title, pathText: p, meaning, fill: i === 0 ? C.paleBlue : C.white, accent: C.blue, titleSize: 17, bodySize: 14 })));
  for (let i = 0; i < 3; i++) {
    link(s, lnodes[i], lnodes[i + 1], i === 0 ? "COPY" : i === 1 ? "AGGREGATE" : "EXPORT", { fromSide: "bottom", toSide: "top", color: C.teal, labelX: 268, labelY: 251 + i * 103 });
    link(s, rnodes[i], rnodes[i + 1], i === 0 ? "TRAIN" : i === 1 ? "API" : "AGGREGATE", { fromSide: "bottom", toSide: "top", color: C.blue, labelX: 920, labelY: 251 + i * 103 });
  }
  callout(s, "frozen-gap", "역사적 LLM 일부는 producer snapshot이 불완전하다. preserved evidence로만 표시하며 fresh regeneration이라고 주장하지 않는다.", 62, 600, 504, 64, C.paleAmber, C.amber, 15);
  callout(s, "clean-gap", "Clean run은 frozen_outputs를 읽지 않는다. API key가 없으면 live call 직전에만 멈추고 이미 만든 upstream 결과는 남긴다.", 714, 600, 504, 64, C.paleRed, C.red, 15);
  notes(s, ["tools/RUN_REPRODUCTION.ps1", "docs/KNOWN_LIMITATIONS.md", "docs/RUN_VERIFICATION.md"]);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
for (const [index, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(qaDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(qaDir, `${stem}.layout.json`), await layout.text());
}
const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(qaDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(outputPath);
