from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS = {"w": W_NS, "r": R_NS, "wp": WP_NS, "a": A_NS, "pic": PIC_NS}
W = f"{{{W_NS}}}"
R = f"{{{R_NS}}}"


NUMBERED_CAPTION_RE = re.compile(
    r"^(?P<kind>표|그림)\s*(?P<number>(?:\d+|[A-Z])(?:[-–—]\d+)?)\s*[\.．]\s*(?P<title>.+?)\s*$"
)
CHAPTER_RE = re.compile(r"^제\s*(?P<number>\d+)\s*장(?:\s+|$)(?P<title>.*)$")
APPENDIX_RE = re.compile(
    r"^부록\s*(?P<number>[A-Z])(?:[\.．]\s*|\s+)(?P<title>.+)$"
)
SECTION_RE = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d+){1,3}|[A-Z]\.\d+(?:\.\d+){0,2}))\s+(?P<title>.+)$"
)

# A numeric token requires at least one digit.  It intentionally captures ranges,
# signs, decimals, commas, percentages and common statistical prefixes while
# leaving the original spelling intact for audit.
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w])(?:[Nn]\s*=\s*)?[+−–-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s*[~–—-]\s*[+−-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)?"
    r"\s*(?:%|％|개|건|회|년|개월|배|점|차원|종|개사|기업|firm-years?|시드|seed)?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParagraphRecord:
    paragraph_index: int
    body_index: int
    pdf_page: int
    printed_page: int | None
    style: str
    text: str
    section: str
    section_number: str
    section_title: str


@dataclass(frozen=True)
class ThesisItem:
    item_id: str
    kind: str
    thesis_number: str
    caption: str
    title: str
    pdf_page: int
    printed_page: int | None
    section: str
    paragraph_index: int
    body_index: int
    body_table_index: int | None
    table_rows: int | None
    table_columns: int | None
    reference_cells: list[list[str]] | None
    reference_role: str
    evidence_class_hint: str


@dataclass(frozen=True)
class NumericClaimCandidate:
    claim_id: str
    paragraph_index: int
    pdf_page: int
    printed_page: int | None
    section: str
    text: str
    numeric_tokens: list[str]
    classification_hint: str
    nearest_item_ids: list[str]
    include_in_ledger: bool
    resolution_status: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest().upper()


def _paragraph_text(element: ET.Element) -> str:
    chunks: list[str] = []
    for node in element.iter():
        local = node.tag.rsplit("}", 1)[-1]
        if local == "t":
            chunks.append(node.text or "")
        elif local in {"tab", "ptab"}:
            chunks.append("\t")
        elif local in {"br", "cr"}:
            chunks.append("\n")
    return "".join(chunks).strip()


def _paragraph_style(element: ET.Element) -> str:
    style = element.find("./w:pPr/w:pStyle", NS)
    return style.get(f"{W}val", "") if style is not None else ""


def _is_toc_style(style: str) -> bool:
    normalized = style.lower().replace(" ", "")
    return "toc" in normalized or normalized.startswith("contents")


def _rendered_page_break_count(element: ET.Element) -> int:
    return len(element.findall(".//w:lastRenderedPageBreak", NS))


def _has_hard_page_break(element: ET.Element) -> bool:
    return any(node.get(f"{W}type", "") == "page" for node in element.findall(".//w:br", NS))


def _heading(text: str, style: str) -> tuple[str, str, str] | None:
    for pattern in (CHAPTER_RE, APPENDIX_RE):
        match = pattern.match(text)
        if match:
            number = match.group("number")
            title = match.group("title").strip()
            return (f"{number} {title}".strip(), number, title)
    # The canonical thesis uses compact numeric style identifiers ("1" and
    # "2") for headings.  Requiring a heading style prevents prose beginning
    # with values such as "0.75 조건에서는 ..." from becoming a false section.
    if style in {"1", "2", "Heading1", "Heading2", "Heading3"}:
        match = SECTION_RE.match(text)
        if match:
            number = match.group("number")
            title = match.group("title").strip()
            return (f"{number} {title}".strip(), number, title)
    return None


def _printed_page(pdf_page: int, first_arabic_pdf_page: int) -> int | None:
    value = pdf_page - first_arabic_pdf_page + 1
    return value if value >= 1 else None


def _table_grid(table: ET.Element) -> list[list[str]]:
    """Extract a rectangular, human-auditable grid from a Word table.

    Horizontal grid spans are represented by the text in the first cell and
    empty continuation cells.  Vertical continuation cells are likewise empty.
    This preserves visible geometry without fabricating duplicate values.
    """

    rows: list[list[str]] = []
    maximum_columns = 0
    for row in table.findall("./w:tr", NS):
        values: list[str] = []
        for cell in row.findall("./w:tc", NS):
            text = "\n".join(
                part for part in (_paragraph_text(p) for p in cell.findall(".//w:p", NS)) if part
            ).strip()
            grid_span = cell.find("./w:tcPr/w:gridSpan", NS)
            span = 1
            if grid_span is not None:
                try:
                    span = max(1, int(grid_span.get(f"{W}val", "1")))
                except ValueError:
                    span = 1
            v_merge = cell.find("./w:tcPr/w:vMerge", NS)
            if v_merge is not None and v_merge.get(f"{W}val", "continue") == "continue":
                text = ""
            values.append(text)
            values.extend([""] * (span - 1))
        maximum_columns = max(maximum_columns, len(values))
        rows.append(values)
    for row in rows:
        row.extend([""] * (maximum_columns - len(row)))
    return rows


def _evidence_class_hint(kind: str, number: str, title: str, section: str) -> str:
    key = f"{kind} {number} {title} {section}".lower()
    design_words = (
        "연구질문",
        "연구가설",
        "구조",
        "흐름",
        "명세",
        "입력변수",
        "행동공간",
        "후보행동",
        "보상",
        "정책조건",
        "하이퍼파라미터",
        "사용 목적",
        "한계",
        "구현 경로",
        "하네스",
    )
    result_words = (
        "결과",
        "정책가치",
        "차이",
        "비교",
        "재현",
        "검정",
        "준수율",
        "기여도",
        "상관",
        "선호",
        "효과",
        "성능",
    )
    has_design = any(word in key for word in design_words)
    has_result = any(word in key for word in result_words)
    if has_design and has_result:
        return "MIXED"
    if has_design or kind == "figure" and number in {"1", "2", "3", "10", "11", "12"}:
        return "NON_EMPIRICAL"
    if has_result:
        return "EMPIRICAL"
    return "MIXED"


def _claim_classification(text: str) -> str:
    empirical = (
        "평균",
        "중앙값",
        "표본",
        "기업",
        "결과",
        "정책가치",
        "신뢰구간",
        "p값",
        "p=",
        "상관",
        "비율",
        "준수율",
        "정확도",
        "auc",
        "spearman",
        "tost",
        "차이",
        "증가",
        "감소",
        "높았",
        "낮았",
    )
    design = (
        "차원",
        "조건",
        "단계",
        "종",
        "상한",
        "가중치",
        "학습률",
        "하이퍼파라미터",
        "seed",
        "시드",
        "연도",
        "기간",
        "프롬프트",
        "정보조건",
    )
    low = text.lower()
    has_empirical = any(token in low for token in empirical)
    has_design = any(token in low for token in design)
    if has_empirical and has_design:
        return "MIXED"
    if has_empirical:
        return "EMPIRICAL"
    if has_design:
        return "NON_EMPIRICAL"
    return "UNCLASSIFIED"


def _is_literature_context(text: str, section_number: str, tokens: list[str]) -> bool:
    if not section_number.startswith("2"):
        return False
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    statistical_cues = ("정책가치", "표본", "본 연구", "본 분석", "실험", "검정", "평균")
    return bool(years) and not any(cue in text for cue in statistical_cues) and (
        len(years) >= 2 or re.search(r"\([12]\d{3}[a-z]?\)", text) is not None
    )


def _nearest_items(
    paragraph: ParagraphRecord,
    items: Iterable[ThesisItem],
    max_page_distance: int = 2,
) -> list[str]:
    candidates = [
        item
        for item in items
        if item.section == paragraph.section
        and abs(item.pdf_page - paragraph.pdf_page) <= max_page_distance
    ]
    candidates.sort(
        key=lambda item: (
            abs(item.pdf_page - paragraph.pdf_page),
            abs(item.paragraph_index - paragraph.paragraph_index),
            item.item_id,
        )
    )
    return [item.item_id for item in candidates[:4]]


def _item_id(kind: str, number: str) -> str:
    compact = number.replace("–", "-").replace("—", "-")
    return ("T" if kind == "table" else "F") + compact


def extract_thesis_inventory(path: Path) -> dict[str, object]:
    """Extract the complete inventory directly from a canonical thesis DOCX.

    The method relies on Word's ``lastRenderedPageBreak`` markers for the page
    occupied by each caption.  The canonical thesis has an Arabic page 1 at PDF
    page 16; this is derived from the first ``제1장`` heading rather than kept as
    a fixed constant.
    """

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))

    body = document.find("w:body", NS)
    if body is None:
        raise ValueError(f"DOCX has no word/document.xml body: {path}")

    page = 1
    paragraph_index = 0
    body_table_index = 0
    section = "FRONT_MATTER"
    section_number = ""
    section_title = ""
    first_arabic_pdf_page: int | None = None
    paragraphs: list[ParagraphRecord] = []
    table_elements: list[tuple[int, int, ET.Element]] = []
    caption_candidates: list[tuple[ParagraphRecord, str, str, str]] = []

    for body_index, element in enumerate(list(body)):
        local = element.tag.rsplit("}", 1)[-1]
        if local == "p":
            paragraph_index += 1
            page += _rendered_page_break_count(element)
            text = _paragraph_text(element)
            style = _paragraph_style(element)
            heading = _heading(text, style) if text and not _is_toc_style(style) else None
            if heading:
                section, section_number, section_title = heading
                if CHAPTER_RE.match(text) and first_arabic_pdf_page is None:
                    first_arabic_pdf_page = page
            record = ParagraphRecord(
                paragraph_index=paragraph_index,
                body_index=body_index,
                pdf_page=page,
                printed_page=None,
                style=style,
                text=text,
                section=section,
                section_number=section_number,
                section_title=section_title,
            )
            paragraphs.append(record)
            if text and not _is_toc_style(style):
                match = NUMBERED_CAPTION_RE.match(text)
                if match:
                    kind = "table" if match.group("kind") == "표" else "figure"
                    caption_candidates.append(
                        (record, kind, match.group("number"), match.group("title").strip())
                    )
            # A hard page break without a lastRenderedPageBreak is uncommon in
            # this thesis.  Word includes the rendered marker in the same run;
            # only count the hard break when no rendered marker exists.
            if _has_hard_page_break(element) and not _rendered_page_break_count(element):
                page += 1
        elif local == "tbl":
            body_table_index += 1
            table_elements.append((body_table_index, body_index, element))

    if first_arabic_pdf_page is None:
        raise ValueError("Could not locate the first numbered chapter in the thesis DOCX")

    # Fill derived printed pages after the Arabic-page offset has been located.
    paragraphs = [
        ParagraphRecord(
            **{
                **asdict(record),
                "printed_page": _printed_page(record.pdf_page, first_arabic_pdf_page),
            }
        )
        for record in paragraphs
    ]
    paragraph_by_body = {record.body_index: record for record in paragraphs}

    tables_by_body = {body_index: (index, table) for index, body_index, table in table_elements}
    items: list[ThesisItem] = []
    used_table_body_indexes: set[int] = set()
    for original_record, kind, number, title in caption_candidates:
        record = paragraph_by_body[original_record.body_index]
        table_index: int | None = None
        table_rows: int | None = None
        table_columns: int | None = None
        reference_cells: list[list[str]] | None = None
        if kind == "table":
            # The body table follows its caption; permit caption notes/blank
            # paragraphs but never cross another caption or section heading.
            for body_index in range(record.body_index + 1, min(record.body_index + 12, len(body))):
                if body_index in tables_by_body:
                    table_index, table_element = tables_by_body[body_index]
                    reference_cells = _table_grid(table_element)
                    table_rows = len(reference_cells)
                    table_columns = max((len(row) for row in reference_cells), default=0)
                    used_table_body_indexes.add(body_index)
                    break
                candidate = paragraph_by_body.get(body_index)
                if candidate and candidate.text:
                    if NUMBERED_CAPTION_RE.match(candidate.text) or _heading(candidate.text, candidate.style):
                        break
        caption = ("표" if kind == "table" else "그림") + f" {number}. {title}"
        items.append(
            ThesisItem(
                item_id=_item_id(kind, number),
                kind=kind,
                thesis_number=number.replace("–", "-").replace("—", "-"),
                caption=caption,
                title=title,
                pdf_page=record.pdf_page,
                printed_page=record.printed_page,
                section=record.section,
                paragraph_index=record.paragraph_index,
                body_index=record.body_index,
                body_table_index=table_index,
                table_rows=table_rows,
                table_columns=table_columns,
                reference_cells=reference_cells,
                reference_role="THESIS_COMPARISON_REFERENCE",
                evidence_class_hint=_evidence_class_hint(kind, number, title, record.section),
            )
        )

    # The canonical Appendix-J implementation-path table is intentionally
    # unnumbered.  It is the only body table without a numbered caption.  Its
    # authority comes from its immediately preceding J.1 heading.
    unused_tables = [
        (index, body_index, table)
        for index, body_index, table in table_elements
        if body_index not in used_table_body_indexes
    ]
    for index, body_index, table in unused_tables:
        preceding = [record for record in paragraphs if record.body_index < body_index and record.text]
        previous = preceding[-1] if preceding else None
        if previous and previous.section_number == "J.1":
            grid = _table_grid(table)
            items.append(
                ThesisItem(
                    item_id="TJ-U",
                    kind="unnumbered_table",
                    thesis_number="J-UNNUMBERED",
                    caption="부록 J.1 무번호 표. 주요 구현 경로",
                    title="주요 구현 경로",
                    pdf_page=previous.pdf_page,
                    printed_page=previous.printed_page,
                    section=previous.section,
                    paragraph_index=previous.paragraph_index,
                    body_index=body_index,
                    body_table_index=index,
                    table_rows=len(grid),
                    table_columns=max((len(row) for row in grid), default=0),
                    reference_cells=grid,
                    reference_role="THESIS_COMPARISON_REFERENCE",
                    evidence_class_hint="NON_EMPIRICAL",
                )
            )

    items.sort(key=lambda item: (item.body_index, item.item_id))

    # Numeric prose claims exclude captions, headings, TOC, references and
    # prompt/code excerpts.  They remain candidates until linked to selected-run
    # calculations or a code/config design source.
    item_paragraphs = {item.paragraph_index for item in items}
    claims: list[NumericClaimCandidate] = []
    for record in paragraphs:
        text = record.text
        if (
            not text
            or record.paragraph_index in item_paragraphs
            or _is_toc_style(record.style)
            or record.style == "ReferenceEntry"
            or _heading(text, record.style)
            or record.section_number.startswith("J.")
        ):
            continue
        tokens = [match.group(0).strip() for match in NUMERIC_TOKEN_RE.finditer(text)]
        if not tokens:
            continue
        literature_context = _is_literature_context(text, record.section_number, tokens)
        claims.append(
            NumericClaimCandidate(
                claim_id=f"NC{len(claims) + 1:04d}",
                paragraph_index=record.paragraph_index,
                pdf_page=record.pdf_page,
                printed_page=record.printed_page,
                section=record.section,
                text=text,
                numeric_tokens=tokens,
                classification_hint=(
                    "LITERATURE_CONTEXT" if literature_context else _claim_classification(text)
                ),
                nearest_item_ids=_nearest_items(record, items),
                include_in_ledger=not literature_context,
                resolution_status=(
                    "EXCLUDED_LITERATURE_CONTEXT"
                    if literature_context
                    else "CANDIDATE_REQUIRES_SELECTED_RUN_OR_CONFIG_EVIDENCE"
                ),
            )
        )

    numbered_tables = [item for item in items if item.kind == "table"]
    figures = [item for item in items if item.kind == "figure"]
    unnumbered_tables = [item for item in items if item.kind == "unnumbered_table"]
    summary = {
        "schema_version": "thesis_docx_inventory_v11",
        "thesis_docx": str(path),
        "thesis_sha256": sha256_file(path),
        "first_arabic_pdf_page": first_arabic_pdf_page,
        "last_rendered_pdf_page_marker": max((p.pdf_page for p in paragraphs), default=0),
        "paragraph_count": len(paragraphs),
        "body_table_count": len(table_elements),
        "numbered_table_count": len(numbered_tables),
        "figure_count": len(figures),
        "unnumbered_table_count": len(unnumbered_tables),
        "terminal_item_count": len(items),
        "numeric_prose_candidate_count": len(claims),
        "inventory_authority": "CANONICAL_THESIS_DOCX_OOXML",
        "printed_value_role": "COMPARISON_REFERENCE_ONLY_NOT_COMPUTE_PARENT",
    }
    return {
        "summary": summary,
        "items": [asdict(item) for item in items],
        "numeric_claim_candidates": [asdict(claim) for claim in claims],
        "paragraphs": [asdict(record) for record in paragraphs],
    }


def validate_canonical_inventory(payload: dict[str, object]) -> list[str]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return ["summary is not an object"]
    errors: list[str] = []
    expected = {
        "numbered_table_count": 56,
        "figure_count": 14,
        "unnumbered_table_count": 1,
        "terminal_item_count": 71,
        "body_table_count": 57,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"{key}: expected {value}, got {summary.get(key)!r}")
    items = payload.get("items", [])
    if isinstance(items, list):
        identifiers = [str(item.get("item_id", "")) for item in items if isinstance(item, dict)]
        duplicates = sorted({value for value in identifiers if value and identifiers.count(value) > 1})
        if duplicates:
            errors.append(f"duplicate item_id values: {duplicates}")
        missing_tables = [
            str(item.get("item_id", ""))
            for item in items
            if isinstance(item, dict)
            and item.get("kind") in {"table", "unnumbered_table"}
            and not item.get("reference_cells")
        ]
        if missing_tables:
            errors.append(f"table cells not associated: {missing_tables}")
    return errors


def dumps_inventory(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
