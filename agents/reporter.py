"""보고서 생성 에이전트 (reporter) — 담당: 이진우

하는 일:
    1) reporter_node  : State에 쌓인 분석 결과로 투자 보고서(마크다운)를 만든다. (LangGraph 노드)
    2) export_report_pdf: 검증이 끝난 보고서를 제출용 PDF로 저장한다. (그래프 실행 후 app.py에서 호출)

핵심 설계 — "숫자는 코드가, 문장은 AI가":
    - 문장(해설): AI(LLM)가 쓴다.
    - 점수표·환산 점수: 코드가 State에서 직접 만들어 끼운다.
    - REFERENCE·[n] 번호: 코드가 State의 sources로 만든다. (AI가 출처를 지어낼 수 없다)
    - 분량·순서·SUMMARY 틀 같은 과제 조건: 코드가 점검하고 어기면 한 번 고쳐 쓰게 한다.

파일 구성(위에서 아래로):
    [A] 프롬프트  [B] 평가표 상수·계산  [C] 점수표  [D] 출처(REFERENCE)
    [E] 과제 조건 점검  [F] 보고서 생성 노드  [G] PDF 저장(대시보드·디자인)
    사실 검증(verifier.py)도 [B]~[D]의 도구를 가져다 쓴다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date
from pathlib import Path

from urllib.parse import urlparse

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import INVEST_THRESHOLD, MAX_REPORT_RETRY, MODEL_ANALYZE
from config import WEIGHTS as TEAM_WEIGHTS

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# [A] 프롬프트 — 문구만 고칠 때는 이 부분만 수정
# ═════════════════════════════════════════════════════════════
# ── 보고서 생성 ──────────────────────────────────────────────
# 시스템 프롬프트: LLM에게 '역할'과 '절대 지킬 규칙'을 알려주는 부분
SYSTEM_PROMPT = """당신은 AI 반도체 스타트업 투자 보고서를 쓰는 애널리스트입니다.
독자는 바쁜 투자위원입니다. 1분 안에 결론과 근거를 파악할 수 있게 쓰세요.
반드시 아래 규칙을 지키세요.
1. 제공된 [분석 자료]에 있는 내용과 숫자만 사용합니다. 자료에 없는 숫자·사실은 절대 만들지 않습니다.
2. 숫자는 자료에 적힌 표기 그대로 옮깁니다. (반올림·단위 변환·추정 금지)
3. 자료에 정보가 없으면 '정보 부족'이라고 씁니다.
4. 출력은 마크다운이며, 아래 [목차]의 순서와 제목('## ' 형식)을 그대로 따릅니다.
5. 맨 앞은 'SUMMARY'입니다. 900자 이내이며, 투자 추천 보고서는 반드시 아래 틀을 그대로 지킵니다.
   **결론:** 결정과 환산 점수를 담은 한 문장
   **핵심 근거**
   - 근거 (정확히 3개)
   **핵심 리스크**
   - 리스크 (정확히 3개)
   **확인 필요**
   - 추가 확인이 필요한 사항 (1~2개)
6. 'REFERENCE' 장은 쓰지 않습니다. (코드가 실제 사용 출처로 자동 첨부합니다.)
7. 본문 전체는 5장(페이지) 이내 분량, 한글 기준 약 8,000자 이내로 간결하게 씁니다.
8. '투자 판단 결과' 장에는 점수표를 직접 쓰지 말고, 표가 들어갈 자리에 {{SCORE_TABLE}} 한 줄만 쓰세요.
   그 아래에 항목별 근거를 문장으로 해설합니다. (표는 코드가 자동으로 채웁니다.)
9. 수치나 사실을 말하는 문장 끝에는 [분석 자료]의 sources_for_citation에 있는 번호를 [1]처럼 붙이세요.
   목록에 없는 번호는 절대 쓰지 말고, 근거 출처가 없으면 번호를 붙이지 않습니다.
"""

# 투자 추천 기업이 있을 때의 목차
TOC_RECOMMEND = """## SUMMARY (위 5번 틀 그대로)
## 1. 사업 아이디어 (핵심 컨셉)
## 2. 시장 규모
## 3. 팀의 구성
## 4. 경쟁 구도
## 5. 투자 판단 결과 (점수표 자리 {{SCORE_TABLE}} + 항목별 근거 해설)
## 6. 사업 리스크 (시장·기술·규제·경쟁)
## 7. 한계점 ('정보 부족' 항목, '확인 필요' 수치, 보류·제외 기업과 사유)"""

# 모든 후보가 보류/제외되었을 때의 목차
TOC_NO_RECOMMEND = """## SUMMARY
## 1. 평가 개요 (평가한 후보와 평가 조건)
## 2. 후보별 평가 결과 (환산 점수와 보류·제외 사유)
## 3. 사업 리스크
## 4. 한계점 및 향후 개선 방향"""

# 사용자 프롬프트 틀: {toc}, {mode}, {materials}, {fix_notes} 자리에 값이 채워진다
USER_TEMPLATE = """[보고서 유형] {mode}

[목차]
{toc}

[분석 자료 - JSON]
{materials}

{fix_notes}
위 자료만 근거로 보고서를 작성하세요."""

# 사실 검증에서 불일치가 나와 '재작성'할 때만 붙는 추가 지시
FIX_NOTES_TEMPLATE = """[수정 요청] 직전 보고서에서 아래 항목이 [분석 자료]와 일치하지 않았습니다.
자료에 있는 값으로 고치거나, 자료에서 확인되지 않으면 그 내용을 빼고 '정보 부족'으로 쓰세요.
{items}
"""

# 과제 조건(분량·순서·SUMMARY 틀) 위반으로 '형식만 고쳐 쓸' 때 붙는 추가 지시
FORMAT_FIX_TEMPLATE = """[형식 수정 요청] 직전 보고서가 아래 과제 조건을 지키지 않았습니다. 내용은 유지하고 형식만 고치세요.
{items}
"""


# ═════════════════════════════════════════════════════════════
# 설정값(상수)
# ═════════════════════════════════════════════════════════════
LLM_MODEL = MODEL_ANALYZE    # 팀 설정(config.py)을 따른다
LLM_TEMPERATURE = 0.0          # 0 = 매번 거의 같은 답(재현성). 사실 기반 글이라 창의성 불필요
MAX_RETRY = MAX_REPORT_RETRY  # 설계 문서: 사실 검증 후 재작성은 최대 1회 (config.py)
MAX_FORMAT_FIXES = 1           # 과제 조건 위반 시 형식만 고쳐 쓰는 횟수(사실 검증 재작성과 별개)

# 평가표(설계서 3장) — 바뀌면 여기만 고친다. 순서는 보고서 표의 표시 순서이기도 하다
# 채점 항목 키는 config.WEIGHTS(영문)를 따르고, 보고서에는 아래 한글 이름으로 표시한다
ITEM_LABELS = {
    "tech": "기술·제품 성숙도",
    "market": "시장성",
    "team": "창업자·팀",
    "traction": "실적·고객 검증",
    "competition": "경쟁 우위",
    "deal": "투자조건",
}
WEIGHTS = {ITEM_LABELS[k]: w for k, w in TEAM_WEIGHTS.items()}  # 한글 이름 → 비중
SCORE_MAX = 5                    # 항목 점수는 1~5점
DECISION_THRESHOLD = INVEST_THRESHOLD  # 설계서 3.4: 환산 점수 70점 이상이면 투자 (config.py)
DECISION_INVEST = "투자"         # 투자 판단 에이전트가 쓰는 결정 값
INSUFFICIENT_MAX_SCORE = 2       # 설계서 3.5: 정보 부족 항목은 최대 2점
INSUFFICIENT_LABEL = "정보 부족"  # 근거 문장에 이 말이 있으면 정보 부족으로 본다
COMPANY_NAME_KEY = "name"        # State의 company 안 기업명 키 (팀원 State와 다르면 여기만 수정)

# 분석 결과가 담긴 State 키들 (설계 문서의 State 표 기준)
ANALYSIS_KEYS = (
    "company", "tech_summary", "market_analysis",
    "competitor_analysis", "team_analysis", "scores",
)

# 과제 조건 점검
SUMMARY_MAX_CHARS = 900     # A4 반 페이지 ≈ 한글 900자 (대략치. 조에서 조정 가능)
PAGE_CHARS = 1800           # A4 한 페이지 ≈ 한글 1,800자 (대략치)
MAX_PAGES = 5               # 과제 조건: 5장 이내
SUMMARY_BULLETS = 3         # SUMMARY 틀: 핵심 근거·핵심 리스크 각 3개
CONFIRM_BULLETS = (1, 2)    # SUMMARY 틀: 확인 필요 1~2개
SUMMARY_LABELS = ("**결론:**", "**핵심 근거**", "**핵심 리스크**", "**확인 필요**")

# 출처 표기 (설계서 5.2) — 종류 → (화면 이름, 필요한 필드, 표기 틀)
SOURCE_FORMATS = {
    "report": ("기관 보고서", ("publisher", "year", "title"),
               "{publisher}({year}). {title}. {url}"),
    "paper": ("학술 논문", ("authors", "year", "title", "journal", "volume_issue", "pages"),
              "{authors}({year}). {title}. {journal}, {volume_issue}, {pages}."),
    "web": ("웹페이지", ("author", "date", "title", "site", "url"),
            "{author}({date}). {title}. {site}, {url}"),
}
OTHER_LABEL = "기타"
UNKNOWN_DATE = "날짜 미상"
REFERENCE_HEADING = "## REFERENCE"

PLACEHOLDER = "{{SCORE_TABLE}}"                      # AI가 쓰는 '표 자리' 표시
SECTION5_PATTERN = re.compile(r"^##\s*5\.[^\n]*\n", re.MULTILINE)
REFERENCE_SECTION_PATTERN = re.compile(r"\n?##\s*REFERENCE.*", re.DOTALL | re.IGNORECASE)
H2_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
TABLE_UNAVAILABLE = "(채점 항목이 누락되어 점수표를 만들 수 없습니다.)"


# ═════════════════════════════════════════════════════════════
# 1) 평가표 계산
# ═════════════════════════════════════════════════════════════
def score_items(scores: dict) -> dict:
    """scores["items"]({영문 키: {score, evidence, ...}})를 보고서용 한글 키로 바꿔 돌려준다."""
    items = scores.get("items") or {}
    return {label: items[key] for key, label in ITEM_LABELS.items() if key in items}


def item_contributions(scores: dict) -> dict[str, float] | None:
    """항목별 '기여 점수' = 점수 ÷ 5 × 비중. 채점 항목이 하나라도 없으면 None."""
    items = score_items(scores)
    if not items:
        return None
    contributions = {}
    for name, weight in WEIGHTS.items():
        try:
            contributions[name] = items[name]["score"] / SCORE_MAX * weight
        except KeyError:
            logger.warning("[채점] 채점 항목 누락: %s", name)
            return None
    return contributions


def recompute_total(scores: dict) -> float | None:
    """항목 점수로 환산 점수를 직접 다시 계산한다. Σ(점수 ÷ 5 × 비중)"""
    contributions = item_contributions(scores)
    return None if contributions is None else round(sum(contributions.values()), 1)


def find_insufficient_items(scores: dict) -> list[str]:
    """'정보 부족'으로 표시된 채점 항목 이름을 찾는다.

    판단 기준(둘 중 하나): items[항목]["insufficient"]가 True이거나, 근거 문장에 '정보 부족'이 있음.
    """
    found = []
    for name, info in score_items(scores).items():
        if info.get("insufficient") or INSUFFICIENT_LABEL in str(info.get("evidence", "")):
            found.append(name)
    return found


# ═════════════════════════════════════════════════════════════
# 2) 점수표 만들기 — 숫자는 코드가
# ═════════════════════════════════════════════════════════════
def build_score_table(scores: dict) -> str:
    """항목별 점수·비중·기여 점수 표와 결정 한 줄을 마크다운으로 만든다."""
    contributions = item_contributions(scores)
    if contributions is None:
        return TABLE_UNAVAILABLE

    total = recompute_total(scores)
    lines = ["| 항목 | 점수 | 비중 | 기여 점수 |", "|---|---|---|---|"]
    for name, weight in WEIGHTS.items():
        score = score_items(scores)[name]["score"]
        lines.append(f"| {name} | {score}/{SCORE_MAX} | {weight}% | {contributions[name]:.1f} |")
    lines.append(f"| **환산 점수** | | 100% | **{total:.1f}** |")
    lines.append("")
    lines.append(f"결정: **{scores.get('decision', '미정')}** "
                 f"(환산 점수 {total:.1f}점, 투자 기준 {DECISION_THRESHOLD}점 이상)")
    return "\n".join(lines)


def insert_score_table(body: str, scores: dict) -> str:
    """본문의 자리표시자를 점수표로 바꾼다. 자리표시자가 없으면 5장 제목 아래에 넣는다."""
    table = build_score_table(scores)
    if PLACEHOLDER in body:
        return body.replace(PLACEHOLDER, table)
    match = SECTION5_PATTERN.search(body)
    if match:
        logger.warning("[표 생성] 자리표시자가 없어 5장 제목 아래에 표를 삽입")
        return body[: match.end()] + "\n" + table + "\n" + body[match.end():]
    logger.warning("[표 생성] 삽입 위치를 찾지 못해 본문 끝에 표를 추가")
    return body + "\n\n" + table


# ═════════════════════════════════════════════════════════════
# 3) 출처(REFERENCE) 만들기 — 출처는 코드가
# ═════════════════════════════════════════════════════════════
def format_source(src: dict) -> tuple[str, str]:
    """출처 하나를 (종류 이름, 표기 문자열)로 바꾼다. 필드가 부족하면 대체 표기를 쓴다."""
    kind = src.get("kind")  # state.Source: "web" | "report"
    site = src.get("site") or urlparse(src.get("url", "")).netloc.removeprefix("www.")
    src = {**src, "year": src.get("year") or (src.get("date") or "")[:4],
           "author": src.get("author") or src.get("publisher") or site,  # 언론사를 모르면 도메인
           "date": src.get("date") or (UNKNOWN_DATE if kind == "web" else ""),  # 게시일 없는 검색 결과
           "site": site, "url": src.get("url", "")}
    if kind in SOURCE_FORMATS:
        label, required, template = SOURCE_FORMATS[kind]
        missing = [f for f in required if not src.get(f)]
        if not missing:
            return label, template.format(**src).strip()
        logger.warning("[REFERENCE] '%s' 출처의 필드 부족 %s → 대체 표기 사용", kind, missing)

    fallback = src.get("citation") or f'{src.get("title", "")} {src.get("url", "")}'.strip()
    label = SOURCE_FORMATS[kind][0] if kind in SOURCE_FORMATS else OTHER_LABEL
    return label, fallback


def collect_sources(state: dict) -> list[dict]:
    """투자 결정 기업의 출처만 모아 중복을 없애고 번호(n)를 매긴다.

    번호는 본문의 [n] 인용과 REFERENCE 목록을 연결하는 열쇠다.
    """
    company_name = (state.get("company") or {}).get(COMPANY_NAME_KEY)
    if not company_name:  # '추천 기업 없음' 보고서에는 출처를 붙이지 않는다
        return []

    collected, seen = [], set()
    for src in state.get("sources", []):
        if src.get("company") != company_name:
            continue
        label, text = format_source(src)
        if not text or text in seen:  # 같은 출처가 여러 노드에서 기록돼도 한 번만
            continue
        seen.add(text)
        collected.append({"n": len(collected) + 1, "label": label, "text": text,
                          "title": src.get("title", "")})
    return collected


def build_reference_section(sources: list[dict]) -> str:
    """REFERENCE 장을 종류별로 묶어 만든다. 번호는 본문 인용 번호와 같다."""
    if not sources:
        return f"\n\n{REFERENCE_HEADING}\n- (사용한 출처 없음)\n"
    labels = [v[0] for v in SOURCE_FORMATS.values()] + [OTHER_LABEL]  # 표시 순서 고정
    parts = [REFERENCE_HEADING]
    for label in labels:
        group = [s for s in sources if s["label"] == label]
        if group:
            parts.append(f"### {label}")
            parts.extend(f'- [{s["n"]}] {s["text"]}' for s in group)
    return "\n\n" + "\n".join(parts) + "\n"


# ═════════════════════════════════════════════════════════════
# 4) 과제 조건 점검
# ═════════════════════════════════════════════════════════════
def _sections(report: str) -> list[tuple[str, str]]:
    """'## 제목' 기준으로 (제목, 내용) 목록을 만든다."""
    matches = list(H2_PATTERN.finditer(report))
    return [(m.group(1), report[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(report)])
            for i, m in enumerate(matches)]


def _count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(("- ", "* ")))


def _summary_structure_violations(summary: str) -> list[str]:
    """투자 추천 SUMMARY가 고정 틀(결론 / 핵심 근거 3 / 핵심 리스크 3 / 확인 필요 1~2)인지 본다."""
    missing = [label for label in SUMMARY_LABELS if label not in summary]
    if missing:
        return [f"SUMMARY 틀에 다음 항목이 없습니다: {', '.join(missing)}"]

    _, rest = summary.split("**핵심 근거**", 1)
    basis, rest = rest.split("**핵심 리스크**", 1)
    risk, confirm = rest.split("**확인 필요**", 1)
    problems = []
    for name, part in (("핵심 근거", basis), ("핵심 리스크", risk)):
        if _count_bullets(part) != SUMMARY_BULLETS:
            problems.append(f"SUMMARY '{name}'는 정확히 {SUMMARY_BULLETS}개여야 합니다(현재 {_count_bullets(part)}개).")
    low, high = CONFIRM_BULLETS
    if not low <= _count_bullets(confirm) <= high:
        problems.append(f"SUMMARY '확인 필요'는 {low}~{high}개여야 합니다(현재 {_count_bullets(confirm)}개).")
    return problems


def check_format(report: str, structured_summary: bool = False) -> list[str]:
    """과제 조건 위반 목록을 돌려준다. 비어 있으면 모두 통과.

    점검: SUMMARY 맨 앞·길이, REFERENCE 맨 뒤, 전체 5장 이내(글자 수 근사),
    structured_summary=True면 SUMMARY 고정 틀까지. (실제 페이지 수는 PDF 저장 때 다시 센다)
    """
    sections = _sections(report)
    if not sections:
        return ["'## ' 제목(장)이 하나도 없습니다. 목차 순서대로 장 제목을 쓰세요."]

    violations = []
    first_title, first_body = sections[0]
    last_title, _ = sections[-1]

    if "SUMMARY" not in first_title.upper():
        violations.append(f"맨 앞 장이 SUMMARY가 아닙니다(현재: '{first_title}').")
    else:
        if len(first_body.strip()) > SUMMARY_MAX_CHARS:
            violations.append(f"SUMMARY가 너무 깁니다({len(first_body.strip())}자). {SUMMARY_MAX_CHARS}자 이내로 줄이세요.")
        if structured_summary:
            violations += _summary_structure_violations(first_body)

    if "REFERENCE" not in last_title.upper():
        violations.append(f"맨 뒤 장이 REFERENCE가 아닙니다(현재: '{last_title}').")

    limit = PAGE_CHARS * MAX_PAGES
    if len(report) > limit:
        violations.append(f"전체 분량이 {len(report)}자로 {MAX_PAGES}장 기준({limit}자)을 넘습니다. 줄이세요.")
    return violations


# ═════════════════════════════════════════════════════════════
# 5) 보고서 생성 노드
# ═════════════════════════════════════════════════════════════
def validate_state(state: dict) -> None:
    """필요한 State 키가 없으면 바로 멈춘다(fail-fast).

    왜: 재료가 빠진 채로 LLM을 부르면 돈만 쓰고 엉터리 보고서가 나온다.
    """
    if _is_recommend_mode(state):
        missing = [k for k in ANALYSIS_KEYS if not state.get(k)]
        if missing:
            raise KeyError(f"보고서 작성에 필요한 State 키가 비어 있습니다: {missing}")


def _is_recommend_mode(state: dict) -> bool:
    """투자 결정 기업이 있는가? (있으면 일반 보고서, 없으면 '추천 기업 없음' 보고서)"""
    return (state.get("scores") or {}).get("decision") == DECISION_INVEST


def build_materials(state: dict, sources: list[dict]) -> dict:
    """LLM에게 줄 '분석 자료' 묶음을 만든다. (자료에 없는 건 LLM이 못 쓰게 하려는 것)"""
    if _is_recommend_mode(state):
        materials = {key: state[key] for key in ANALYSIS_KEYS}
        # LLM이 [n]을 붙일 수 있도록 '번호 붙은 출처 목록'을 보여 준다(표기 원문은 코드가 관리)
        materials["sources_for_citation"] = [
            {"n": s["n"], "type": s["label"], "title": s["title"]} for s in sources
        ]
    else:
        materials = {}
    materials["rejected"] = state.get("rejected", [])  # 보류·제외 기업과 사유는 한계점에 항상 쓰인다
    return materials


def assemble_report(state: dict, body: str, sources: list[dict]) -> str:
    """AI가 쓴 본문에 코드가 만든 부분(점수표, REFERENCE)을 합쳐 완성본을 만든다."""
    body = REFERENCE_SECTION_PATTERN.sub("", body).rstrip()  # AI가 쓴 REFERENCE는 지운다(진짜는 코드가)
    if _is_recommend_mode(state):
        body = insert_score_table(body, state["scores"])
    else:
        body = body.replace(PLACEHOLDER, "")  # 추천 없음 보고서에는 점수표가 없다
    return body + build_reference_section(sources)


def _get_llm():
    """LLM 객체를 만든다. API 키가 없으면 여기서 바로 알려준다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    from langchain.chat_models import init_chat_model  # 필요할 때만 import (테스트 시 없어도 되게)

    return init_chat_model(LLM_MODEL, model_provider="openai", temperature=LLM_TEMPERATURE)


def _verify_fix_notes(state: dict) -> str:
    """사실 검증이 불일치를 찾아 되돌려 보냈다면, 그 목록을 프롬프트에 붙인다."""
    result = state.get("verify_result") or {}
    mismatches = result.get("mismatches", [])
    if result.get("passed", True) or not mismatches:
        return ""
    items = "\n".join(f'- "{m["value"]}" (문맥: {m["context"]})' for m in mismatches)
    return FIX_NOTES_TEMPLATE.format(items=items)


def reporter_node(state: dict, llm=None) -> dict:
    """보고서 생성 노드. State를 받아 '바꿀 키만' dict로 돌려준다.

    LangGraph 규칙: 노드는 State 전체가 아니라 갱신할 키만 반환하면 된다.
    llm 인자는 테스트 때 가짜 LLM을 끼우려는 용도(평소엔 None).
    """
    logger.info("[보고서 생성] 시작")
    validate_state(state)

    recommend = _is_recommend_mode(state)
    mode = "투자 추천 보고서" if recommend else "추천 기업 없음 보고서"
    logger.info("[보고서 생성] 유형: %s", mode)

    # 재작성 여부: 직전 검증이 실패로 끝났다면 재작성 → 횟수 +1
    prev = state.get("verify_result") or {}
    is_rewrite = bool(prev) and not prev.get("passed", True)
    retry_count = state.get("retry_count", 0) + (1 if is_rewrite else 0)
    if retry_count > MAX_RETRY:
        raise ValueError(f"재작성 횟수({retry_count})가 한도({MAX_RETRY})를 넘었습니다.")
    logger.info("[보고서 생성] 재작성 여부=%s, retry_count=%d", is_rewrite, retry_count)

    sources = collect_sources(state) if recommend else []
    logger.info("[보고서 생성] 사용 가능한 출처 %d건", len(sources))
    materials = json.dumps(build_materials(state, sources), ensure_ascii=False, indent=2)
    llm = llm or _get_llm()

    verify_notes, format_notes, report = _verify_fix_notes(state), "", ""
    for attempt in range(MAX_FORMAT_FIXES + 1):
        user_prompt = USER_TEMPLATE.format(
            mode=mode, toc=TOC_RECOMMEND if recommend else TOC_NO_RECOMMEND,
            materials=materials, fix_notes=verify_notes + format_notes,
        )
        logger.info("[보고서 생성] LLM 호출 (시도 %d)", attempt + 1)
        response = llm.invoke([("system", SYSTEM_PROMPT), ("user", user_prompt)])
        report = assemble_report(state, response.content.strip(), sources)

        violations = check_format(report, structured_summary=recommend)  # 과제 조건을 코드로 점검
        if not violations:
            break
        logger.warning("[보고서 생성] 과제 조건 위반 %d건: %s", len(violations), violations)
        format_notes = FORMAT_FIX_TEMPLATE.format(items="\n".join(f"- {v}" for v in violations))
    else:
        logger.warning("[보고서 생성] 형식 수정 후에도 위반이 남았습니다. 현재 본문을 유지합니다.")

    logger.info("[보고서 생성] 완료 (%d자)", len(report))
    return {"report": report, "retry_count": retry_count}


# ═════════════════════════════════════════════════════════════
# [G] PDF 저장 — 1쪽: 제목 밴드 → SUMMARY → 결정 대시보드, 이후 본문·REFERENCE
#     한글 글꼴: 환경변수 KOREAN_FONT_PATH → fonts/Pretendard-Regular.ttf → 시스템 글꼴 → 내장 글꼴 순으로 찾는다.
#     (reportlab은 TrueType(.ttf)만 쓸 수 있고 .otf/CFF 글꼴은 못 쓴다)
# ═════════════════════════════════════════════════════════════
# ── 설정값(상수) ─────────────────────────────────────────────
CAMPUS = "울산캠퍼스"                            # 설계서 파일명 기준. 조가 바뀌면 여기만 수정
CLASS_NO = "4반"
MEMBERS = ["강지수", "김다은", "이진우", "이채목", "장정훈"]
OUTPUT_DIR = Path("outputs")                   # 결과 저장 폴더 (설계서 디렉터리 구조의 outputs/)
FONT_ENV_VAR = "KOREAN_FONT_PATH"
FONT_NAME, FONT_NAME_BOLD = "KoreanFont", "KoreanFont-Bold"
CID_FALLBACK_FONT = "HYGothic-Medium"
FONT_CANDIDATES = [
    Path("fonts/Pretendard-Regular.ttf"),   # 저장소에 포함 (SIL OFL)
    Path("fonts/NanumGothic.ttf"),
    Path("/Library/Fonts/NanumGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),   # macOS
    Path("C:/Windows/Fonts/malgun.ttf"),                          # Windows
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),      # Linux
]

# 색 팔레트 — 투자=초록, 보류=주황, 확인 필요=빨강, 기본=남색
NAVY, INK, MUTED = "#0b3d91", "#1a1a1a", "#6b7280"
LINE, PANEL, NOTE_BG = "#d8dee9", "#f3f6fb", "#f5f5f5"
GOOD, HOLD, BAD = "#1f8a4c", "#d97706", "#c0392b"

# 레이아웃
PAGE_MARGIN = 16 * mm
BOTTOM_EXTRA = 4 * mm                          # 바닥글 자리
USABLE_PAGE_HEIGHT = A4[1] - 2 * PAGE_MARGIN - BOTTOM_EXTRA   # 한 쪽의 본문 높이
SUMMARY_MAX_PAGE_RATIO = 0.5                   # 과제 조건: SUMMARY는 1/2 페이지 이내
BODY_SIZE, SMALL_SIZE = 9.5, 8
DASH_HEIGHT = 168                              # 대시보드 카드·차트 높이(pt)
CARD_RATIO = 0.42                              # 대시보드에서 왼쪽 카드가 차지하는 폭 비율
SCORE_BAR_W, SCORE_BAR_H = 70, 9
MAX_REJECTED_ROWS = 8                          # '추천 없음' 보고서의 후보 막대 최대 개수
SCORE_TABLE_FRACS = (0.36, 0.26, 0.14, 0.24)   # 점수표(4칸) 열 폭 비율

# State 안의 company 키 후보 (팀원 State와 다르면 여기만 수정)
SECTOR_KEYS = ("segment", "sector", "세부_분야", "세부분야", "domain")
STAGE_KEYS = ("stage", "투자_단계", "투자단계", "round")

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
SCORE_CELL_PATTERN = re.compile(r"^(\d)/(\d)$")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
NOTE_TITLE = "자동 검증 결과"


# ── 파일명 ───────────────────────────────────────────────────
def build_output_filename(campus: str = CAMPUS, class_no: str = CLASS_NO,
                          members: list[str] | None = None) -> str:
    """과제 파일명 규칙: RAG-Output_{캠퍼스}-{X반}_{이름1+이름2+...}.pdf"""
    members = members or MEMBERS
    if not members:
        raise ValueError("조원 이름 목록이 비어 있습니다.")
    return f"RAG-Output_{campus}-{class_no}_{'+'.join(members)}.pdf"


# ── 글꼴 ─────────────────────────────────────────────────────
def _find_font_files() -> list[Path]:
    """글꼴 후보를 우선순위 순서로 돌려준다."""
    candidates = [Path(os.environ[FONT_ENV_VAR])] if os.environ.get(FONT_ENV_VAR) else []
    return [p for p in candidates + FONT_CANDIDATES if p.exists()]


def register_korean_font() -> tuple[str, str]:
    """한글 글꼴을 등록하고 (보통 글꼴 이름, 굵은 글꼴 이름)을 돌려준다."""
    for path in _find_font_files():
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
        except (TTFError, OSError) as err:
            logger.warning("[PDF] 글꼴 사용 불가(%s): %s", path, err)
            continue
        bold_stem = path.stem.replace("Regular", "Bold") if "Regular" in path.stem else path.stem + "Bold"
        bold_path, bold_name = path.with_name(bold_stem + path.suffix), FONT_NAME
        if bold_path.exists():  # 굵은 글꼴이 같은 폴더에 있으면 쓰고, 없으면 보통 글꼴로 대신한다
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, str(bold_path)))
                bold_name = FONT_NAME_BOLD
            except (TTFError, OSError) as err:
                logger.warning("[PDF] 굵은 글꼴 사용 불가(%s): %s", bold_path, err)
        pdfmetrics.registerFontFamily(FONT_NAME, normal=FONT_NAME, bold=bold_name,
                                      italic=FONT_NAME, boldItalic=bold_name)
        logger.info("[PDF] 한글 글꼴 사용: %s", path)
        return FONT_NAME, bold_name

    logger.warning("[PDF] 한글 TrueType 글꼴을 찾지 못해 내장 글꼴(%s)을 사용합니다. "
                   "fonts/Pretendard-Regular.ttf를 넣으면 더 안정적입니다.", CID_FALLBACK_FONT)
    pdfmetrics.registerFont(UnicodeCIDFont(CID_FALLBACK_FONT))
    pdfmetrics.registerFontFamily(CID_FALLBACK_FONT, normal=CID_FALLBACK_FONT, bold=CID_FALLBACK_FONT,
                                  italic=CID_FALLBACK_FONT, boldItalic=CID_FALLBACK_FONT)
    return CID_FALLBACK_FONT, CID_FALLBACK_FONT


# ── 스타일 ───────────────────────────────────────────────────
def _hex(code: str):
    return colors.HexColor(code)


def _inline(text: str) -> str:
    """한 줄 안의 서식을 PDF용으로 바꾼다. (&,<,> 이스케이프 후 **굵게** 처리)"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _styles(font: str, bold: str, size: float = BODY_SIZE) -> dict[str, ParagraphStyle]:
    """본문 크기(size)에 맞춘 문단 스타일 묶음."""
    base = dict(fontName=font, alignment=TA_LEFT, textColor=_hex(INK))
    lead = size * 1.6
    return {
        "body": ParagraphStyle("body", fontSize=size, leading=lead, spaceAfter=3, **base),
        "bullet": ParagraphStyle("bullet", fontSize=size, leading=lead, leftIndent=11, bulletIndent=2, **base),
        "sub": ParagraphStyle("sub", fontSize=size - 0.5, leading=lead - 1, leftIndent=24, bulletIndent=14,
                              **{**base, "textColor": _hex(MUTED)}),
        "label": ParagraphStyle("label", fontSize=size, leading=lead, spaceBefore=3,
                                **{**base, "fontName": bold, "textColor": _hex(NAVY)}),
        "h1": ParagraphStyle("h1", fontSize=16, leading=22, spaceAfter=6, **{**base, "fontName": bold}),
        "h2": ParagraphStyle("h2", fontSize=12.5, leading=17, **{**base, "fontName": bold, "textColor": _hex(NAVY)}),
        "h3": ParagraphStyle("h3", fontSize=size + 1, leading=(size + 1) * 1.4, spaceBefore=4, spaceAfter=2,
                             **{**base, "fontName": bold}),
        "cell": ParagraphStyle("cell", fontSize=size - 0.5, leading=(size - 0.5) * 1.5, **base),
    }


# ── 작은 그래픽: 점수 막대 ───────────────────────────────────
def _score_color(score: int) -> str:
    return GOOD if score >= 4 else NAVY if score == 3 else HOLD


def _score_bar(score: int, max_score: int, font: str) -> Drawing:
    """표 안에 들어가는 작은 점수 막대. (예: 4/5 → 80% 채워진 막대와 '4/5' 글자)"""
    drawing = Drawing(SCORE_BAR_W + 26, SCORE_BAR_H + 2)
    drawing.add(Rect(0, 1, SCORE_BAR_W, SCORE_BAR_H, fillColor=_hex(LINE), strokeColor=None))
    drawing.add(Rect(0, 1, SCORE_BAR_W * score / max_score, SCORE_BAR_H,
                     fillColor=_hex(_score_color(score)), strokeColor=None))
    drawing.add(String(SCORE_BAR_W + 5, 2, f"{score}/{max_score}", fontName=font, fontSize=8, fillColor=_hex(INK)))
    return drawing


# ── 마크다운 → PDF 조각 ──────────────────────────────────────
def _table(rows: list[str], styles: dict, width: float, font: str) -> Table:
    """마크다운 표(| a | b |) 줄들을 PDF 표로 바꾼다. 'N/5' 칸은 점수 막대로 바꾼다."""
    parsed = []
    for line in rows:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):  # |---|---| 구분선은 건너뜀
            continue
        row = []
        for cell in cells:
            m = SCORE_CELL_PATTERN.match(cell)
            row.append(_score_bar(int(m.group(1)), int(m.group(2)), font) if m
                       else Paragraph(_inline(cell), styles["cell"]))
        parsed.append(row)

    col_count = max(len(r) for r in parsed)
    for r in parsed:  # 칸 수가 모자란 줄은 빈 칸으로 채운다
        r.extend([Paragraph("", styles["cell"])] * (col_count - len(r)))
    fracs = SCORE_TABLE_FRACS if col_count == 4 else [1 / col_count] * col_count
    table = Table(parsed, colWidths=[width * f for f in fracs], repeatRows=1)

    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _hex(LINE)),
        ("BACKGROUND", (0, 0), (-1, 0), _hex(PANEL)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    last_cells = [c for c in rows[-1].split("|")]
    if any("환산 점수" in c for c in last_cells):  # 합계 행은 강조
        style.append(("BACKGROUND", (0, len(parsed) - 1), (-1, len(parsed) - 1), _hex(PANEL)))
        style.append(("LINEABOVE", (0, len(parsed) - 1), (-1, len(parsed) - 1), 1, _hex(NAVY)))
    table.setStyle(TableStyle(style))
    return table


def content_flowables(lines: list[str], styles: dict, width: float, font: str) -> list:
    """제목을 뺀 내용 줄들을 PDF 조각(표·목록·문단)으로 바꾼다."""
    flow, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):  # 표는 연속된 '|' 줄을 한 덩어리로 모은다
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            flow += [_table(block, styles, width, font), Spacer(1, 4)]
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*:?", line.strip()):      # '**핵심 근거**' 같은 소제목 줄
            flow.append(Paragraph(_inline(line.strip()), styles["label"]))
        elif line.startswith("  - "):
            flow.append(Paragraph(_inline(line[4:]), styles["sub"], bulletText="–"))
        elif line.startswith(("- ", "* ")):
            flow.append(Paragraph(_inline(line[2:]), styles["bullet"], bulletText="•"))
        elif line.strip():
            flow.append(Paragraph(_inline(line.strip()), styles["body"]))
        i += 1
    return flow


def parse_blocks(markdown: str) -> list[dict]:
    """마크다운을 제목 기준 덩어리 [{level, title, lines}]로 나눈다. level 0 = 첫 제목 앞 글."""
    blocks = [{"level": 0, "title": "", "lines": []}]
    for line in markdown.splitlines():
        m = HEADING_PATTERN.match(line)
        if m:
            blocks.append({"level": len(m.group(1)), "title": m.group(2).strip(), "lines": []})
        else:
            blocks[-1]["lines"].append(line)
    return blocks


def _heading2(title: str, styles: dict, width: float) -> Table:
    """장 제목: 좌측 색 띠가 있는 한 줄."""
    box = Table([[Paragraph(_inline(title), styles["h2"])]], colWidths=[width])
    box.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, 0), 3.5, _hex(NAVY)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7),
                             ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return box


def _panel(flowables: list, width: float, background: str, bar: str | None) -> Table:
    """색 배경 박스(SUMMARY, 검증 결과에 사용). bar가 있으면 왼쪽에 색 띠."""
    box = Table([[flowables]], colWidths=[width])
    style = [("BACKGROUND", (0, 0), (-1, -1), _hex(background)),
             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if bar:
        style.append(("LINEBEFORE", (0, 0), (0, 0), 3.5, _hex(bar)))
    box.setStyle(TableStyle(style))
    return box


def render_blocks(blocks: list[dict], font: str, bold: str, width: float, has_dashboard: bool) -> list:
    """덩어리별로 디자인을 달리해 PDF 조각 목록을 만든다."""
    normal, small = _styles(font, bold), _styles(font, bold, SMALL_SIZE)
    flow = []
    for block in blocks:
        level, title, lines = block["level"], block["title"], block["lines"]
        upper = title.upper()
        if level == 0:
            flow += content_flowables(lines, normal, width, font)
        elif level == 1:
            if not has_dashboard:  # 대시보드가 제목 밴드를 대신하므로 그때는 건너뛴다
                flow.append(Paragraph(_inline(title), normal["h1"]))
            flow += content_flowables(lines, normal, width, font)
        elif level == 2 and "SUMMARY" in upper:
            inner = [Paragraph("SUMMARY", normal["h2"])] + content_flowables(lines, normal, width - 24, font)
            flow += [Spacer(1, 4), _panel(inner, width, PANEL, NAVY)]
        elif level == 2 and "REFERENCE" in upper:
            flow += [Spacer(1, 8), _heading2(title, small, width), Spacer(1, 3)]
            flow += content_flowables(lines, small, width, font)
        elif level == 2:
            body = content_flowables(lines, normal, width, font)
            flow += [Spacer(1, 7), KeepTogether([_heading2(title, normal, width), Spacer(1, 3)] + body[:2])]
            flow += body[2:]
        elif level == 3 and title.startswith(NOTE_TITLE):
            inner = [Paragraph(_inline(title), small["label"])] + content_flowables(lines, small, width - 24, font)
            flow += [Spacer(1, 6), _panel(inner, width, NOTE_BG, MUTED)]
        elif level == 3:
            flow.append(Paragraph(_inline(title), small["h3"] if "REFERENCE" in upper else normal["h3"]))
            flow += content_flowables(lines, small if _is_reference_child(blocks, block) else normal, width, font)
    return flow


def _is_reference_child(blocks: list[dict], block: dict) -> bool:
    """이 소제목(###)이 REFERENCE 장 안에 있는가? (그렇다면 작은 글씨로 보여 준다)"""
    parent = ""
    for b in blocks:
        if b["level"] == 2:
            parent = b["title"].upper()
        if b is block:
            return "REFERENCE" in parent
    return False


# ── 결정 대시보드 (1쪽 위쪽) ─────────────────────────────────
def _first_value(data: dict, keys: tuple[str, ...]) -> str:
    return next((str(data[k]) for k in keys if data.get(k)), "")


def _title_band(company: dict, decision_text: str, report_date: str, styles: dict, font: str,
                bold: str, width: float) -> Table:
    """1쪽 맨 위 남색 제목 밴드."""
    name = company.get(COMPANY_NAME_KEY)
    title = f"{name} 투자 평가 보고서" if name else "투자 평가 보고서"   # 추천 없음이면 부제에 '추천 기업 없음'이 표시된다
    title_style = ParagraphStyle("band_t", fontName=bold, fontSize=17, leading=23, textColor=colors.white)
    sub_style = ParagraphStyle("band_s", fontName=font, fontSize=8.5, leading=12, textColor=_hex("#c9d6f2"))
    meta = " · ".join(v for v in (_first_value(company, SECTOR_KEYS), _first_value(company, STAGE_KEYS),
                                  decision_text, report_date) if v)
    band = Table([[Paragraph(_inline(title), title_style)],
                  [Paragraph(_inline(meta), sub_style)]], colWidths=[width])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(NAVY)),
                              ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (0, 0), 9),
                              ("BOTTOMPADDING", (0, 1), (0, 1), 9), ("TOPPADDING", (0, 1), (0, 1), 0),
                              ("BOTTOMPADDING", (0, 0), (0, 0), 1)]))
    return band


def _score_card(total: float, decision: str, font: str, bold: str, w: float, h: float = DASH_HEIGHT) -> Drawing:
    """결정 배지 + 큰 환산 점수 + 70점 기준선이 있는 게이지."""
    d = Drawing(w, h)
    color = GOOD if decision == DECISION_INVEST else HOLD
    d.add(Rect(0, h - 30, 74, 24, rx=12, ry=12, fillColor=_hex(color), strokeColor=None))
    d.add(String(37, h - 22, decision, fontName=bold, fontSize=12, fillColor=colors.white, textAnchor="middle"))

    score_text = f"{total:.1f}"
    d.add(String(0, h - 86, score_text, fontName=bold, fontSize=42, fillColor=_hex(NAVY)))
    text_w = pdfmetrics.stringWidth(score_text, bold, 42)
    d.add(String(text_w + 5, h - 86, "/ 100", fontName=font, fontSize=12, fillColor=_hex(MUTED)))

    gauge_w, gauge_y = w - 14, h - 118
    d.add(Rect(0, gauge_y, gauge_w, 10, fillColor=_hex(LINE), strokeColor=None))
    d.add(Rect(0, gauge_y, gauge_w * min(total, 100) / 100, 10, fillColor=_hex(color), strokeColor=None))
    x = gauge_w * DECISION_THRESHOLD / 100  # 투자 기준선
    d.add(Line(x, gauge_y - 4, x, gauge_y + 14, strokeColor=_hex(NAVY), strokeWidth=1.6))
    d.add(String(x, gauge_y + 18, f"기준 {DECISION_THRESHOLD}", fontName=font, fontSize=7.5,
                 fillColor=_hex(NAVY), textAnchor="middle"))
    d.add(String(0, 6, "환산 점수 = Σ(항목 점수 ÷ 5 × 비중)", fontName=font, fontSize=7, fillColor=_hex(MUTED)))
    return d


def _radar(items: dict, font: str, bold: str, w: float, h: float = DASH_HEIGHT) -> Drawing:
    """6개 평가 항목 점수를 한눈에 보는 레이더 차트."""
    names = list(WEIGHTS)
    scores = [items[n]["score"] for n in names]
    cx, cy, radius = w / 2, h / 2 - 2, min(w / 2 - 66, h / 2 - 20)
    angle = lambda i: math.pi / 2 - 2 * math.pi * i / len(names)          # 12시 방향에서 시계 방향
    point = lambda i, r: (cx + r * math.cos(angle(i)), cy + r * math.sin(angle(i)))

    d = Drawing(w, h)
    for ring in range(1, SCORE_MAX + 1):  # 1~5점 눈금 다각형
        pts = [c for i in range(len(names)) for c in point(i, radius * ring / SCORE_MAX)]
        d.add(Polygon(pts, fillColor=None, strokeColor=_hex(LINE), strokeWidth=0.6))
    for i in range(len(names)):  # 축
        x, y = point(i, radius)
        d.add(Line(cx, cy, x, y, strokeColor=_hex(LINE), strokeWidth=0.6))

    data_pts = [c for i, s in enumerate(scores) for c in point(i, radius * s / SCORE_MAX)]
    d.add(Polygon(data_pts, fillColor=_hex(NAVY), fillOpacity=0.22, strokeColor=_hex(NAVY), strokeWidth=1.6))
    for i, s in enumerate(scores):
        x, y = point(i, radius * s / SCORE_MAX)
        d.add(Circle(x, y, 2.3, fillColor=_hex(NAVY), strokeColor=None))

    for i, (name, s) in enumerate(zip(names, scores)):  # 항목 이름과 점수
        x, y = point(i, radius + 9)
        cos, sin = math.cos(angle(i)), math.sin(angle(i))
        anchor = "middle" if abs(cos) < 0.3 else "start" if cos > 0 else "end"
        y += 4 if sin > 0.5 else -10 if sin < -0.5 else -3
        d.add(String(x, y, f"{name} {s}", fontName=font, fontSize=7.5, fillColor=_hex(INK), textAnchor=anchor))
    return d


def _rejected_bars(rejected: list[dict], font: str, bold: str, w: float) -> Drawing | None:
    """'추천 기업 없음' 보고서용: 후보별 환산 점수 막대와 70점 기준선."""
    rows = [r for r in rejected if isinstance(r.get("total"), (int, float))][:MAX_REJECTED_ROWS]
    if not rows:
        return None
    row_h, label_w = 22, 90
    d = Drawing(w, row_h * len(rows) + 16)
    bar_w = w - label_w - 40
    top = row_h * len(rows)
    for i, r in enumerate(rows):
        y = top - (i + 1) * row_h + 6
        d.add(String(0, y + 2, str(r.get("company", "?"))[:10], fontName=font, fontSize=8.5, fillColor=_hex(INK)))
        d.add(Rect(label_w, y, bar_w, 10, fillColor=_hex(LINE), strokeColor=None))
        d.add(Rect(label_w, y, bar_w * min(r["total"], 100) / 100, 10, fillColor=_hex(HOLD), strokeColor=None))
        d.add(String(label_w + bar_w + 6, y + 2, f'{r["total"]:g}', fontName=bold, fontSize=8.5,
                     fillColor=_hex(INK)))
    x = label_w + bar_w * DECISION_THRESHOLD / 100
    d.add(Line(x, 2, x, top + 4, strokeColor=_hex(NAVY), strokeWidth=1.4))
    d.add(String(x, top + 6, f"투자 기준 {DECISION_THRESHOLD}", fontName=font, fontSize=7.5,
                 fillColor=_hex(NAVY), textAnchor="middle"))
    return d


def _badge_strip(state: dict, report: str, styles: dict, width: float) -> Table | None:
    """검증 배지 줄: 수치 검증 / 원문 대조(RAG) / (있으면) 주장 검증(LLM) / 인용 / 정보 부족."""
    result = state.get("verify_result")
    badges = []  # (제목, 값, 색)
    if result:
        problems = result.get("mismatches", [])
        is_rule = lambda m: not str(m.get("kind", "")).startswith(("claim", "rag"))
        bad_numbers = sum(1 for m in problems if m.get("kind") == "number")
        checked = result.get("checked", 0)
        value = f"{max(checked - bad_numbers, 0)}/{checked} 일치"
        others = sum(1 for m in problems if is_rule(m) and m.get("kind") != "number")
        if others:
            value += f" · 표기·인용 {others}건"
        badges.append(("수치 검증(State)", value, GOOD if not [m for m in problems if is_rule(m)] else BAD))

        rag = result.get("rag") or {}
        if rag.get("checked"):
            rag_bad = sum(1 for m in problems if str(m.get("kind", "")).startswith("rag"))
            found = rag["checked"] - rag.get("not_found", 0)
            badges.append(("원문 대조(RAG)", f"{found}/{rag['checked']} 원문 확인 · 불일치 {rag_bad}",
                           GOOD if not rag_bad else BAD))
        else:
            badges.append(("원문 대조(RAG)", "미실행", MUTED))

        if result.get("judged"):
            claims_bad = sum(1 for m in problems if m.get("kind") == "claim")
            badges.append(("주장 검증(LLM)", f"{result['judged'] - claims_bad}/{result['judged']} 근거 확인",
                           GOOD if not claims_bad else HOLD))
    else:
        badges.append(("수치 검증", "미검증", MUTED))
    body = report.split("## REFERENCE")[0]
    badges.append(("인용", f"{len(set(CITATION_PATTERN.findall(body)))}건 연결", NAVY))
    lacking = len(find_insufficient_items(state.get("scores") or {}))
    badges.append(("정보 부족", f"{lacking}개 항목" if lacking else "없음", HOLD if lacking else GOOD))

    cells = [[Paragraph(f'<font size="7.5" color="{MUTED}">{title}</font><br/>'
                        f'<font color="{color}"><b>{_inline(value)}</b></font>', styles["cell"])
              for title, value, color in badges]]
    strip = Table(cells, colWidths=[width / len(badges)] * len(badges))
    style = [("BACKGROUND", (0, 0), (-1, -1), _hex(PANEL)),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 9)]
    for col, (_, _, color) in enumerate(badges):
        style.append(("LINEBEFORE", (col, 0), (col, 0), 3, _hex(color)))
    strip.setStyle(TableStyle(style))
    return strip


def build_dashboard(state: dict, report: str, report_date: str, font: str, bold: str,
                    width: float) -> tuple[list, list]:
    """1쪽 대시보드를 (제목 밴드, 시각 자료) 두 부분으로 만든다.

    왜 둘로 나누나: 과제 조건상 SUMMARY가 맨 앞이어야 하므로,
    순서를 '제목 밴드 → SUMMARY → 시각 자료(배지·게이지·레이더)'로 끼워 넣기 위해서다.
    필요한 State가 없으면 ([], []) = 대시보드 생략.
    """
    scores, company = state.get("scores") or {}, state.get("company") or {}
    normal = _styles(font, bold)
    total, items = recompute_total(scores), score_items(scores)

    if total is not None and items:  # 투자 추천 보고서: 배지 + 게이지 + 레이더 + 검증 배지
        decision = scores.get("decision", "미정")
        card_w = width * CARD_RATIO
        row = Table([[_score_card(total, decision, font, bold, card_w - 8),
                      _radar(items, font, bold, width - card_w)]], colWidths=[card_w, width - card_w])
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        header = [_title_band(company, decision, report_date, normal, font, bold, width), Spacer(1, 4)]
        visuals = [Spacer(1, 10), row, Spacer(1, 6), _badge_strip(state, report, normal, width), Spacer(1, 4)]
        return header, visuals

    # 추천 기업 없음: 제목 밴드 + 후보별 점수 막대
    header = [_title_band({}, "추천 기업 없음", report_date, normal, font, bold, width), Spacer(1, 4)]
    bars = _rejected_bars(state.get("rejected") or [], font, bold, width)
    return header, ([Spacer(1, 10), bars, Spacer(1, 4)] if bars else [])


def _split_at_summary(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """덩어리 목록을 'SUMMARY까지'와 '그 뒤'로 나눈다. SUMMARY가 없으면 (전부, [])."""
    for i, block in enumerate(blocks):
        if block["level"] == 2 and "SUMMARY" in block["title"].upper():
            return blocks[: i + 1], blocks[i + 1:]
    return blocks, []


def _flow_height(flowables: list, width: float) -> float:
    """PDF 조각들이 실제로 차지할 세로 높이(pt)를 잰다. (글자 수 추정이 아니라 실제 조판 결과)"""
    return sum(f.wrap(width, A4[1])[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in flowables)


def measure_summary_ratio(report: str) -> float:
    """SUMMARY 박스가 한 쪽 본문 높이의 몇 %를 차지하는지 잰다. (과제 조건: 1/2 페이지 이내)"""
    font, bold = register_korean_font()
    width = A4[0] - 2 * PAGE_MARGIN
    summary_blocks = [b for b in parse_blocks(report)
                      if b["level"] == 2 and "SUMMARY" in b["title"].upper()]
    if not summary_blocks:
        return 0.0
    height = _flow_height(render_blocks(summary_blocks, font, bold, width, True), width)
    return height / USABLE_PAGE_HEIGHT


# ── 쪽 머리글·바닥글 ─────────────────────────────────────────
def _page_decorations(font: str, label: str, counter: dict):
    """쪽마다 호출되는 그리기 함수 2개(첫 쪽용, 이후 쪽용)를 만든다. 쪽 수도 함께 센다."""
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(_hex(LINE))
        canvas.line(PAGE_MARGIN, 11 * mm, A4[0] - PAGE_MARGIN, 11 * mm)
        canvas.setFont(font, 7.5)
        canvas.setFillColor(_hex(MUTED))
        canvas.drawString(PAGE_MARGIN, 7 * mm, label)
        canvas.drawRightString(A4[0] - PAGE_MARGIN, 7 * mm, f"{doc.page}쪽")
        canvas.restoreState()
        counter["n"] += 1

    def later(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(_hex(MUTED))
        canvas.drawString(PAGE_MARGIN, A4[1] - 10 * mm, label)
        canvas.setStrokeColor(_hex(NAVY))
        canvas.setLineWidth(1.2)
        canvas.line(PAGE_MARGIN, A4[1] - 11.5 * mm, A4[0] - PAGE_MARGIN, A4[1] - 11.5 * mm)
        canvas.restoreState()
        footer(canvas, doc)

    return footer, later


# ── PDF 저장 ─────────────────────────────────────────────────
def export_report_pdf(report: str, output_dir: Path = OUTPUT_DIR, filename: str | None = None,
                      state: dict | None = None, report_date: str | None = None) -> tuple[Path, int]:
    """보고서를 PDF로 저장하고 (저장 경로, 페이지 수)를 돌려준다. 5장을 넘으면 경고한다.

    state를 함께 주면 1쪽에 '제목 밴드 → SUMMARY → 결정 대시보드' 순서로 그린다. (없으면 본문만)
    SUMMARY가 실제 조판 기준으로 반 페이지를 넘으면 경고한다.
    """
    if not report or not report.strip():
        raise ValueError("빈 보고서는 PDF로 저장할 수 없습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (filename or build_output_filename())
    report_date = report_date or date.today().isoformat()

    font, bold = register_korean_font()
    width = A4[0] - 2 * PAGE_MARGIN
    state = state or {}

    # 순서: 제목 밴드 → SUMMARY → 시각 자료(대시보드) → 1장~ → REFERENCE
    header, visuals = (build_dashboard(state, report, report_date, font, bold, width)
                       if state.get("scores") else ([], []))
    until_summary, after_summary = _split_at_summary(parse_blocks(report))
    has_band = bool(header)
    story = (header + render_blocks(until_summary, font, bold, width, has_band)
             + visuals + render_blocks(after_summary, font, bold, width, has_band))

    ratio = measure_summary_ratio(report)  # SUMMARY가 반 페이지를 넘는지 실제 높이로 확인
    if ratio > SUMMARY_MAX_PAGE_RATIO:
        logger.warning("[PDF] SUMMARY가 한 쪽의 %.0f%%를 차지해 반 페이지 기준을 넘었습니다.", ratio * 100)
    else:
        logger.info("[PDF] SUMMARY 높이: 한 쪽의 %.0f%%", ratio * 100)

    # 추천 기업이 없으면 마지막으로 평가한 기업 이름을 바닥글에 넣지 않는다
    company_name = (state.get("company") or {}).get(COMPANY_NAME_KEY, "") if _is_recommend_mode(state) else ""
    label = " · ".join(v for v in ("AI 스타트업 투자 평가", company_name) if v)
    counter = {"n": 0}
    first, later = _page_decorations(font, label, counter)

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
                            topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN + BOTTOM_EXTRA, title=path.stem)
    doc.build(story, onFirstPage=first, onLaterPages=later)

    pages = counter["n"]
    if pages > MAX_PAGES:
        logger.warning("[PDF] %d장으로 과제 기준(%d장 이내)을 넘었습니다. 본문을 줄이세요.", pages, MAX_PAGES)
    else:
        logger.info("[PDF] 저장 완료: %s (%d장)", path, pages)
    return path, pages
