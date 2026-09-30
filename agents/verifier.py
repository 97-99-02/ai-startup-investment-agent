"""사실 검증 에이전트 (verifier) — 담당: 이진우

하는 일: 보고서를 출처에 따라 나눠 검증한다. (LangGraph 노드 verifier_node)
    1단계 — State 대조(코드, 비용 0, 결과가 매번 같음)
        - 수치를 '값 + 단위'로 읽어 State의 수치와 비교 ('120억 원' ≠ '120억 달러')
          인터넷에서 온 값(투자금·팀·경쟁사)은 State에 수집한 수치가 곧 원문이므로 여기서 끝난다.
        - 한글로 쓴 금액('천억 원'), 자료에 없는 과장 표현('업계 1위')
        - 본문 [n] 번호가 REFERENCE에 있는지, 환산 점수 재계산이 맞는지
    2단계 — RAG 원문 대조(문서에서 온 기술·시장 내용, Self-RAG 방식)
        - 검색 → 관련성 검사 → (관련 없으면) 검색어 재작성 후 재검색 → 관련 조각으로만 근거성 검사
        - 관련 원문을 끝내 못 찾은 문장은 틀렸다고 하지 않고 '판정 제외'(엉뚱한 조각 때문의 오탐 방지)
          수치가 원문에 있는지(코드) + 원문과 모순되지 않는지(LLM 검수)
        - 분석 에이전트가 문서를 잘못 옮겨 State가 틀린 경우도 잡는다.
        - 벡터 DB에 연결할 수 없으면 건너뛰고, 건너뛴 사실을 한계점에 적는다.
    3단계 — LLM 검수(선택): 인용이 달린 나머지 서술 주장이 State로 뒷받침되는지
        (투자 판단 에이전트 judge.py와는 다른 것. 여기서는 '보고서 문장 검수'만 한다)
    불일치면 보고서 생성으로 1회 되돌리고, 그래도 안 맞으면 '(확인 필요)' 표시 후 한계점에 기록하고 종료한다.

검증 범위(정직하게):
    - 인터넷 출처 값은 분석 에이전트가 수집한 값과의 일치만 본다.
    - 원문 대조는 검색된 문서 조각 범위 안에서만 유효하다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from agents.reporter import (
    DECISION_THRESHOLD,
    INSUFFICIENT_MAX_SCORE,
    REFERENCE_HEADING,
    SCORE_MAX,
    WEIGHTS,
    _is_recommend_mode,
    collect_sources,
    find_insufficient_items,
    item_contributions,
    recompute_total,
    score_items,
)

from config import MAX_REPORT_RETRY, MODEL_ANALYZE

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# 프롬프트 (LLM 검수용)
# ═════════════════════════════════════════════════════════════
# ── 사실 검증: LLM 검수 ─────────────────────────────────────
JUDGE_SYSTEM = """당신은 투자 보고서의 사실 검증관입니다.
[분석 자료]와 보고서 [주장] 목록을 비교해 각 주장이 자료로 뒷받침되는지 판정합니다.
- supported: 자료에 같은 내용이 있거나 자료로부터 직접 도출된다.
- unsupported: 자료에 없거나, 자료와 모순되거나, 자료보다 과장되었다.
자료에 없는 배경지식으로 판단하지 마세요. 애매하면 unsupported로 판정하세요.
반드시 아래 JSON 형식으로만 응답하세요(다른 텍스트 없이):
{"results": [{"id": 1, "verdict": "supported", "reason": "한 문장 근거"}]}"""

JUDGE_USER_TEMPLATE = """[분석 자료 - JSON]
{materials}

[주장 목록]
{claims}
"""

# ── 사실 검증: RAG 원문 대조 Judge ───────────────────────────
# 분석 자료(State)가 아니라 '원래 문서에서 다시 찾은 조각'을 기준으로 판정한다.
RAG_JUDGE_SYSTEM = """당신은 투자 보고서의 원문 대조 검증관입니다.
각 [주장]과 그 아래 [원문 조각]을 비교해 판정합니다.
- supported: 원문 조각에 같은 내용이 있거나 직접 도출된다.
- unsupported: 원문 조각과 수치·사실이 다르거나, 원문보다 과장되었다.
- not_found: 원문 조각에 이 주장과 관련된 내용이 아예 없다. (인터넷에서 온 정보일 수 있으므로 틀렸다고 보지 않는다)
원문 조각에 없는 배경지식으로 판단하지 마세요.
반드시 아래 JSON 형식으로만 응답하세요(다른 텍스트 없이):
{"results": [{"id": 1, "verdict": "supported", "reason": "한 문장 근거"}]}"""

# Self-RAG ② 관련성 검사: 검색해 온 조각이 문장과 관련 있는가
RELEVANCE_SYSTEM = """당신은 검색 결과의 관련성 평가자입니다.
각 [주장]에 대해, 아래 [조각] 중 그 주장의 사실(수치·기업·기술·시장)을 확인하는 데 쓸 수 있는 조각의 번호만 고르세요.
주제만 비슷하고 주장을 확인할 수 없는 조각은 고르지 마세요. 쓸 수 있는 조각이 없으면 빈 목록을 주세요.
반드시 아래 JSON 형식으로만 응답하세요(다른 텍스트 없이):
{"results": [{"id": 1, "relevant": [1, 3]}]}"""

# Self-RAG ③ 검색어 재작성: 관련 조각을 못 찾은 주장의 검색어를 고친다
REWRITE_SYSTEM = """당신은 문서 검색어 작성자입니다.
각 [주장]을 원문 문서에서 찾기 쉬운 짧은 검색어(핵심 명사 3~6개)로 바꾸세요. 숫자는 빼고 주제어를 남기세요.
반드시 아래 JSON 형식으로만 응답하세요(다른 텍스트 없이):
{"results": [{"id": 1, "query": "AI 반도체 시장 규모 성장률 전망"}]}"""

CLAIMS_WITH_CHUNKS_TEMPLATE = """{blocks}
"""

RAG_JUDGE_USER_TEMPLATE = """아래 주장마다 원문 조각을 보고 판정하세요.

{blocks}
"""


# ── 설정값(상수) ─────────────────────────────────────────────
MAX_RETRY = MAX_REPORT_RETRY         # 설계 문서: 재작성 1회 (config.py)
SCORE_TOLERANCE = 0.05              # 환산 점수 재계산 허용 오차
YEAR_RANGE = range(1990, 2101)      # 이 범위의 정수는 '연도'로 보고 검사 제외
MIN_BARE_VALUE = 10                 # 단위 없는 수는 이 값 이상만 검사(목차 번호·표의 4/5 등을 거르려는 것)
CONTEXT_CHARS = 25                  # 불일치 앞뒤로 보여줄 글자 수
VALUE_PRECISION = 6                 # 값 비교 시 반올림 자릿수(1,200억 = 1.2e11 같은 부동소수 오차 방지)
MARK_UNVERIFIED = "(확인 필요)"      # 끝내 안 맞는 표현 뒤에 붙일 표시
MARK_BAD_CITATION = "[출처 확인 필요]"  # 존재하지 않는 인용 번호를 바꿔 넣을 표시
NOTE_HEADING = "### 자동 검증 결과"    # 한계점 뒤에 붙는 검증 요약 제목
MARKABLE_KINDS = ("number", "spelled", "overclaim", "citation", "rag_number")  # 본문에 표시할 불일치 종류

# 수량 단위: 배수(억·조…)와 단위(원·달러·%…)
MAGNITUDES = {"조": 1e12, "억": 1e8, "만": 1e4, "천": 1e3}
UNITS = ("달러", "원", "%", "배", "명", "개", "건", "년", "개월", "점", "nm", "TOPS", "W")
YEAR_UNIT, SCORE_UNIT = "년", "점"
# 10 미만이어도 검사할 단위: 비율·배수·점수·금액·성능. 개수 단위(개·건·명·개월)는 '6개 항목', '3건' 같은
# 서술에 흔히 쓰여 오탐이 많으므로 단위 없는 수와 같은 기준(10 이상)만 검사한다.
ALWAYS_CHECK_UNITS = ("%", "배", "점", "원", "달러", "nm", "TOPS", "W")

# 자료에 없으면 '과장'으로 보는 표현 (자료에 같은 표현이 있으면 허용)
# ('유일'처럼 짧은 말은 '유일하게 부족하다' 같은 일반 문장까지 잡으므로 넣지 않는다)
OVERCLAIM_WORDS = ("업계 1위", "국내 1위", "세계 1위", "세계 최초", "국내 최초", "국내 유일", "세계 유일",
                   "유일한 기업", "독보적", "압도", "독점", "최고 수준", "최대 규모")

# LLM 검수 설정 (투자 판단 judge.py와 무관)
USE_LLM_JUDGE = True                # False면 Judge 단계를 항상 건너뜀
JUDGE_MODEL = MODEL_ANALYZE           # 팀 설정(config.py)을 따른다
MAX_JUDGE_CLAIMS = 8                # 한 번에 판정할 주장 수 상한(비용·오탐 관리)
MAX_NOTED_CLAIMS = 3                # 한계점에 적을 미확인 주장 수 상한

# RAG 원문 대조 설정 — 팀원(기술 요약·시장성 평가)의 벡터 DB 설정과 맞출 것
USE_RAG_CHECK = True                            # False면 원문 대조를 항상 건너뜀
RAG_TOP_K = 3                                   # 문장 하나당 가져올 원문 조각 수
MAX_RAG_CLAIMS = 6                              # 원문 대조할 문장 수 상한(검색·Judge 비용 관리)
MIN_CLAIM_CHARS = 8                             # 이보다 짧은 문장은 대조하지 않음
MAX_QUERY_REWRITES = 1                          # Self-RAG: 관련 조각이 없을 때 검색어를 고쳐 다시 찾는 횟수
MIN_KEYWORD_OVERLAP = 2                         # LLM 없이 관련성을 볼 때, 겹쳐야 하는 핵심 단어 수
MIN_KEYWORD_CHARS = 2                           # 핵심 단어로 볼 최소 글자 수
KOREAN_ENDINGS = ("으로", "에서", "이며", "이다", "했다", "한다", "된다", "였다", "라는", "에는",
                  "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "다")   # 단어 끝 조사·어미
RAG_SECTION_KEYWORDS = ("사업 아이디어", "시장 규모")   # 문서(RAG)에서 온 내용이 들어가는 장
RAG_STATE_KEYS = ("tech_summary", "market_analysis")  # 문서(RAG)에서 온 State 칸
WEB_STATE_KEYS = ("company", "competitor_analysis", "team_analysis")  # 인터넷에서 온 State 칸
SOURCE_META_KEYS = ("source", "title", "file_name")   # 원문 조각의 문서명 메타데이터 후보
PAGE_META_KEYS = ("page", "page_number")              # 원문 조각의 쪽 번호 메타데이터 후보

EVIDENCE_KEYS = (
    "company", "tech_summary", "market_analysis",
    "competitor_analysis", "team_analysis", "scores", "rejected",
)

# 수량 규칙: 숫자 + (배수) + (단위).  예) '1,200억 원', '25%', '4점', '300억 달러', '15년'
QUANTITY_PATTERN = re.compile(
    r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(" + "|".join(MAGNITUDES) + r")?\s*("
    + "|".join(sorted(UNITS, key=len, reverse=True)) + r")?"
)
# 금액을 한글 숫자로 쓴 경우: '천억 원', '삼백억 달러'
SPELLED_PATTERN = re.compile(r"[일이삼사오육칠팔구십백천]+\s*(?:조|억|만)\s*(?:원|달러)")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")            # [1] [12] — 수치 검사 때는 먼저 지운다
SENTENCE_SPLIT = re.compile(r"(?<=[.다])\s+")           # 문장 나누기(마침표·'다' 뒤 공백)


# ═════════════════════════════════════════════════════════════
# 수량 읽기
# ═════════════════════════════════════════════════════════════
def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def parse_quantities(text: str) -> list[dict]:
    """글에서 수량을 뽑는다. 각 항목: value(배수 반영한 값), unit(단위, 없으면 ''),
    scaled(억·조 등 배수가 붙었는지), surface(글에 쓰인 모양), pos(위치)."""
    found = []
    for m in QUANTITY_PATTERN.finditer(text):
        number = _to_float(m.group(1))
        if number is None:
            continue
        magnitude, unit = m.group(2), m.group(3) or ""
        value = round(number * MAGNITUDES.get(magnitude, 1), VALUE_PRECISION)
        found.append({"value": value, "unit": unit, "scaled": bool(magnitude),
                      "surface": m.group(0).strip(), "pos": m.start()})
    return found


# ═════════════════════════════════════════════════════════════
# 정답지 만들기
# ═════════════════════════════════════════════════════════════
def _without_tech_trace(data: dict) -> dict:
    """청크 ID·URL 등 추적용 메타데이터를 수치 정답지에서 제외한다."""
    tech = data.get("tech_summary")
    if isinstance(tech, dict):
        return {**data, "tech_summary": {k: v for k, v in tech.items() if k != "evidence"}}
    return data


def _evidence_json(state: dict) -> str:
    return json.dumps(_without_tech_trace({k: state.get(k) for k in EVIDENCE_KEYS}),
                      ensure_ascii=False)


def _body_of(report: str) -> str:
    """REFERENCE 앞까지의 본문. (출처 목록은 검사 대상이 아니다)"""
    return report.split(REFERENCE_HEADING)[0]


def build_evidence(state: dict) -> dict:
    """보고서에 써도 되는 수량의 '정답지'를 만든다.

    반환:
        pairs  : {(값, 단위)} — 단위가 있는 수는 값과 단위가 둘 다 맞아야 통과
        values : {값}          — 단위를 생략한 수는 값만 맞으면 통과
        points : {값}          — '~점'으로 쓴 수(점수)는 평가표 값과 대조
    """
    quantities = parse_quantities(_evidence_json(state))
    # 출처의 근거 원문(snippet)도 정답지다 (state.Source: 사실 검증에 사용). 투자 기업의 출처만 쓴다
    name = (state.get("company") or {}).get("name")
    snippets = " ".join(str(src.get("snippet", "")) for src in state.get("sources", [])
                        if src.get("company") == name)
    quantities += parse_quantities(snippets)
    pairs = {(q["value"], q["unit"]) for q in quantities}
    values = {q["value"] for q in quantities}

    # 설계서 고정값: 항목 비중(%), 합계 100%, 투자 기준 70점
    for weight in list(WEIGHTS.values()) + [sum(WEIGHTS.values())]:
        pairs.add((float(weight), "%"))
        values.add(float(weight))
    values.add(float(DECISION_THRESHOLD))

    # 점수로 쓸 수 있는 값: 항목 점수, 기여 점수, 환산 점수, 기준·만점·정보 부족 상한
    scores = state.get("scores") or {}
    points = {float(DECISION_THRESHOLD), float(SCORE_MAX), float(INSUFFICIENT_MAX_SCORE)}
    points |= {float(info["score"]) for info in score_items(scores).values() if "score" in info}
    contributions = item_contributions(scores)
    if contributions is not None:
        rounded = {round(v, 1) for v in contributions.values()}
        points |= rounded
        values |= rounded
    # 보류·제외된 후보의 환산 점수도 보고서('추천 기업 없음' 등)에 '58점'처럼 쓰인다
    rejected_totals = [r.get("total") for r in (state.get("rejected") or []) if isinstance(r, dict)]
    for total in [recompute_total(scores), scores.get("total")] + rejected_totals:
        if isinstance(total, (int, float)):
            points.add(round(float(total), 1))
            values.add(round(float(total), 1))
    return {"pairs": pairs, "values": values, "points": points}


# ═════════════════════════════════════════════════════════════
# 1단계: 규칙 검증
# ═════════════════════════════════════════════════════════════
def _context(body: str, pos: int) -> str:
    return body[max(0, pos - CONTEXT_CHARS): pos + CONTEXT_CHARS].replace("\n", " ")


def _needs_check(q: dict) -> bool:
    """검사 대상인가? 연도와 단위 없는 작은 수(목차 번호, 표의 '4/5')는 제외한다."""
    is_year = q["value"] == int(q["value"]) and int(q["value"]) in YEAR_RANGE
    if q["unit"] == YEAR_UNIT and is_year:
        return False
    if q["scaled"] or q["unit"] in ALWAYS_CHECK_UNITS:
        return True                       # 억·조 배수나 %·배·점·금액 단위는 작은 수라도 검사 (예: 8%, 3배, 2점)
    return not is_year and q["value"] >= MIN_BARE_VALUE   # 개수 단위·단위 없음: 10 이상만


def _is_supported(q: dict, evidence: dict) -> bool:
    """이 수량이 정답지에 있는가?"""
    if q["unit"] == SCORE_UNIT:                        # '80점', '4점' → 평가표 값과 대조
        return q["value"] in evidence["points"]
    if q["unit"]:                                      # '120억 원', '25%' → 값과 단위 모두 일치
        return (q["value"], q["unit"]) in evidence["pairs"]
    return q["value"] in evidence["values"]            # 단위 생략 → 값만 일치하면 인정


def find_quantity_mismatches(report: str, evidence: dict) -> tuple[list[dict], int]:
    """보고서 수량 중 정답지에 없는 것을 찾는다. (불일치 목록, 검사한 수량 개수)"""
    body = CITATION_PATTERN.sub("", _body_of(report))   # [12] 같은 인용 번호는 수치 검사에서 제외
    mismatches, checked = [], 0
    for q in parse_quantities(body):
        if not _needs_check(q):
            continue
        checked += 1
        if not _is_supported(q, evidence):
            mismatches.append({"kind": "number", "value": q["surface"], "surface": q["surface"],
                               "context": _context(body, q["pos"])})
    return mismatches, checked


def find_wording_problems(report: str, state: dict) -> list[dict]:
    """한글로 쓴 금액('천억 원')과 자료에 없는 과장 표현('업계 1위')을 찾는다."""
    body = _body_of(report)
    evidence_text = _evidence_json(state)
    problems = []
    for m in SPELLED_PATTERN.finditer(body):
        problems.append({"kind": "spelled", "value": m.group(0), "surface": m.group(0),
                         "context": "금액을 한글로 표기함(자료의 숫자 표기로 써야 대조 가능) · "
                                    + _context(body, m.start())})
    for word in OVERCLAIM_WORDS:
        pos = body.find(word)
        if pos >= 0 and word not in evidence_text:
            problems.append({"kind": "overclaim", "value": word, "surface": word,
                             "context": "자료에 없는 과장 표현 · " + _context(body, pos)})
    return problems


def check_citations(report: str, sources: list[dict]) -> list[dict]:
    """본문 [n] 중 REFERENCE에 없는 번호를 찾는다. (AI가 출처를 지어낸 경우를 잡는다)"""
    body = _body_of(report)
    valid = {s["n"] for s in sources}
    problems = []
    for n in sorted({int(m.group(1)) for m in CITATION_PATTERN.finditer(body)} - valid):
        problems.append({"kind": "citation", "value": f"[{n}]", "surface": f"[{n}]",
                         "context": "REFERENCE에 없는 출처 번호 · " + _context(body, body.find(f"[{n}]"))})
    return problems


def check_total_score(scores: dict) -> list[dict]:
    """State의 환산 점수가 항목 점수 재계산값과 다르면 알린다(투자 판단 쪽 계산 실수 방지)."""
    recomputed, stated = recompute_total(scores), scores.get("total")
    if recomputed is None or stated is None:
        return []
    if abs(recomputed - float(stated)) > SCORE_TOLERANCE:
        return [{"kind": "score", "value": str(stated),
                 "context": f"State 환산 점수 {stated} ≠ 항목 점수로 재계산한 {recomputed}"}]
    return []


# ═════════════════════════════════════════════════════════════
# 3단계: LLM 검수
# ═════════════════════════════════════════════════════════════
def _section_sentences(report: str):
    """본문을 (장 제목, 문장)으로 하나씩 내보낸다. 제목·표 줄은 건너뛴다."""
    section = ""
    for line in _body_of(report).splitlines():
        text = line.strip()
        if text.startswith("## "):
            section = text[3:]
            continue
        if not text or text.startswith(("#", "|")):
            continue
        if text.startswith(("- ", "* ")):
            text = text[2:]
        for sentence in SENTENCE_SPLIT.split(text.replace("**", "")):
            if sentence.strip():
                yield section, sentence.strip()


def _is_rag_section(section: str) -> bool:
    return any(keyword in section for keyword in RAG_SECTION_KEYWORDS)


def extract_claims(report: str, skip_rag_sections: bool = False) -> list[str]:
    """Judge에게 맡길 '인용이 달린 핵심 주장 문장'을 뽑는다. (표·제목은 제외, 상한 적용)

    skip_rag_sections=True면 원문 대조(RAG)가 맡는 장은 빼서 같은 문장을 두 번 판정하지 않는다.
    """
    claims = [sentence for section, sentence in _section_sentences(report)
              if CITATION_PATTERN.search(sentence) and not (skip_rag_sections and _is_rag_section(section))]
    return claims[:MAX_JUDGE_CLAIMS]


def _resolve_judge(judge_llm):
    """Judge LLM을 정한다. False=끄기, 객체=그것 사용, None=자동(키가 있으면 실제 LLM, 없으면 건너뜀)."""
    if judge_llm is False or not USE_LLM_JUDGE:
        return None
    if judge_llm is not None:
        return judge_llm
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("[사실 검증] OPENAI_API_KEY가 없어 LLM 검수 단계를 건너뜁니다.")
        return None
    from langchain.chat_models import init_chat_model  # 필요할 때만 import

    return init_chat_model(JUDGE_MODEL, model_provider="openai", temperature=0)


def parse_judge_response(content: str) -> list[dict]:
    """Judge 응답(JSON)을 읽는다. 코드블록(```)이 붙어 와도 처리한다."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    return json.loads(text.strip())["results"]


def judge_claims(report: str, state: dict, judge_llm, skip_rag_sections: bool = False) -> tuple[list[dict], int]:
    """핵심 주장을 Judge LLM이 판정한다. (근거 없음 목록, 판정한 주장 수) 반환.

    응답을 읽지 못하면 경고만 남기고 건너뛴다. 검증 보조 단계가 전체를 멈추면 안 되기 때문이다.
    """
    claims = extract_claims(report, skip_rag_sections)
    if not claims:
        return [], 0
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    user_prompt = JUDGE_USER_TEMPLATE.format(materials=_evidence_json(state), claims=numbered)
    logger.info("[사실 검증] LLM 검수 호출 (주장 %d건)", len(claims))
    response = judge_llm.invoke([("system", JUDGE_SYSTEM), ("user", user_prompt)])
    try:
        results = parse_judge_response(response.content)
    except (ValueError, KeyError, TypeError, IndexError) as err:  # JSONDecodeError는 ValueError의 하위 클래스
        logger.warning("[사실 검증] LLM 검수 응답을 읽지 못해 건너뜁니다: %s", err)
        return [], 0

    unsupported = []
    for item in results:
        try:
            index, verdict = int(item["id"]), item["verdict"]
        except (KeyError, TypeError, ValueError):
            continue
        if verdict == "unsupported" and 1 <= index <= len(claims):
            unsupported.append({"kind": "claim", "value": claims[index - 1][:60],
                                "context": str(item.get("reason", "자료에서 뒷받침을 찾지 못함"))})
    return unsupported, len(claims)


# ═════════════════════════════════════════════════════════════
# 2단계: RAG 원문 대조 — 문서에서 온 기술·시장 내용을 '원래 문서'와 비교
# ═════════════════════════════════════════════════════════════
def load_rag_retriever(company_name: str | None = None):
    """팀 RAG 기반(rag/retriever.py)의 벡터 DB를 열어 검색기를 만든다. 열 수 없으면 None(= 원문 대조 건너뜀).

    기술·시장 문서를 함께 검색한다(doc_type 필터 없음). 벡터 DB는 `uv run python -m rag.ingest`로 만든다.
    """
    try:  # 무거운 라이브러리라 필요할 때만 불러온다
        from rag.retriever import get_vectorstore
        store = get_vectorstore()
    except (FileNotFoundError, ImportError) as err:
        logger.info("[원문 대조] 벡터 DB를 열 수 없어 건너뜁니다: %s", err)
        return None
    logger.info("[원문 대조] 벡터 DB 연결")
    return store.as_retriever(search_kwargs={"k": RAG_TOP_K})


def _resolve_retriever(retriever, company_name: str | None):
    """검색기를 정한다. False=끄기, 객체=그것 사용, None=자동(벡터 DB가 있으면 연결)."""
    if retriever is False or not USE_RAG_CHECK:
        return None
    return retriever if retriever is not None else load_rag_retriever(company_name)


def _doc_label(doc) -> str:
    """원문 조각의 출처 이름표. 예) 'KISDI 보고서 p.42'"""
    meta = getattr(doc, "metadata", {}) or {}
    name = next((str(meta[k]) for k in SOURCE_META_KEYS if meta.get(k)), "원문")
    name = Path(name).stem if "/" in name or "." in name else name
    page = next((meta[k] for k in PAGE_META_KEYS if meta.get(k) is not None), None)
    return f"{name} p.{page}" if page is not None else name


def extract_rag_claims(report: str) -> list[str]:
    """원문 대조할 문장: '사업 아이디어'·'시장 규모' 장의 문장 (인용 번호는 떼고, 상한 적용)."""
    claims = [CITATION_PATTERN.sub("", sentence).strip()
              for section, sentence in _section_sentences(report) if _is_rag_section(section)]
    return [c for c in claims if len(c) >= MIN_CLAIM_CHARS][:MAX_RAG_CLAIMS]


def _pairs_of(data) -> set:
    return {(q["value"], q["unit"]) for q in
            parse_quantities(json.dumps(_without_tech_trace(data), ensure_ascii=False))}


def _found_in(q: dict, pairs: set) -> bool:
    """수량 q가 (값, 단위) 모음에 있는가? 단위를 생략했으면 값만 맞아도 인정."""
    if q["unit"]:
        return (q["value"], q["unit"]) in pairs
    return any(value == q["value"] for value, _ in pairs)


def _keywords(text: str) -> set[str]:
    """문장의 핵심 단어 모음(숫자·기호 제거, 조사·어미 떼기). LLM 없이 관련성을 볼 때 쓴다."""
    words = set()
    for token in re.findall(r"[가-힣A-Za-z]+", text):
        for ending in KOREAN_ENDINGS:
            if token.endswith(ending) and len(token) - len(ending) >= MIN_KEYWORD_CHARS:
                token = token[: -len(ending)]
                break
        if len(token) >= MIN_KEYWORD_CHARS and token not in KOREAN_ENDINGS:   # '이다' 같은 어미 자체는 제외
            words.add(token.lower())
    return words


def _ask_json(llm, system: str, blocks: list[str]) -> list[dict]:
    """LLM에 한 번 묻고 JSON 결과 목록을 받는다. 읽지 못하면 빈 목록(= 이 단계 결과 없음)."""
    response = llm.invoke([("system", system),
                           ("user", CLAIMS_WITH_CHUNKS_TEMPLATE.format(blocks="\n\n".join(blocks)))])
    try:
        return parse_judge_response(response.content)
    except (ValueError, KeyError, TypeError, IndexError) as err:  # JSONDecodeError는 ValueError의 하위 클래스
        logger.warning("[원문 대조] LLM 응답을 읽지 못했습니다: %s", err)
        return []


def _chunk_text(doc) -> str:
    return getattr(doc, "page_content", str(doc))


def grade_relevance(claims: dict[int, str], docs: dict[int, list], llm) -> dict[int, list]:
    """Self-RAG ② 관련성 검사: 주장마다 '확인에 쓸 수 있는 조각'만 남긴다.

    LLM이 있으면 LLM이 고르고, 없으면 핵심 단어가 MIN_KEYWORD_OVERLAP개 이상 겹치는 조각만 남긴다.
    """
    if llm is None:
        return {i: [d for d in docs[i]
                    if len(_keywords(claims[i]) & _keywords(_chunk_text(d))) >= MIN_KEYWORD_OVERLAP]
                for i in claims}
    blocks = [f"[주장 {i}] {claims[i]}\n" + "\n".join(f"  [조각 {j}] {_chunk_text(d)}"
                                                  for j, d in enumerate(docs[i], 1)) for i in claims]
    kept = {i: [] for i in claims}
    for item in _ask_json(llm, RELEVANCE_SYSTEM, blocks):
        try:
            i, picks = int(item["id"]), item.get("relevant", [])
        except (KeyError, TypeError, ValueError):
            continue
        if i in docs:
            kept[i] = [docs[i][j - 1] for j in picks if isinstance(j, int) and 1 <= j <= len(docs[i])]
    return kept


def rewrite_queries(claims: dict[int, str], llm) -> dict[int, str]:
    """Self-RAG ③ 검색어 재작성: 관련 조각을 못 찾은 주장을 짧은 검색어로 바꾼다.

    LLM이 없으면 숫자를 빼고 핵심 단어만 이어 붙인다.
    """
    if llm is None:
        return {i: " ".join(sorted(_keywords(c))) for i, c in claims.items()}
    queries = {i: " ".join(sorted(_keywords(c))) for i, c in claims.items()}   # 응답이 없을 때의 대비
    for item in _ask_json(llm, REWRITE_SYSTEM, [f"[주장 {i}] {c}" for i, c in claims.items()]):
        try:
            i, query = int(item["id"]), str(item["query"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if i in queries and query:
            queries[i] = query
    return queries


def _search(retriever, query: str) -> list:
    return list(retriever.invoke(query))[:RAG_TOP_K]


def retrieve_relevant(claims: dict[int, str], retriever, llm) -> tuple[dict[int, list], int]:
    """Self-RAG ①~③: 검색 → 관련성 검사 → (관련 없으면) 검색어 재작성 후 재검색·재검사.

    반환: (주장별 관련 조각, 검색어를 고친 주장 수)
    """
    docs = {i: _search(retriever, c) for i, c in claims.items()}                  # ① 검색
    relevant = grade_relevance(claims, docs, llm)                                # ② 관련성 검사
    rewritten = 0
    for _ in range(MAX_QUERY_REWRITES):
        missing = {i: claims[i] for i in claims if not relevant[i]}
        if not missing:
            break
        queries = rewrite_queries(missing, llm)                                   # ③ 검색어 재작성
        retry_docs = {i: _search(retriever, queries[i]) for i in missing}
        retry_relevant = grade_relevance(missing, retry_docs, llm)
        for i in missing:
            relevant[i] = retry_relevant[i]
        rewritten += len(missing)
        logger.info("[원문 대조] 관련 조각이 없던 %d건의 검색어를 고쳐 다시 검색", len(missing))
    return relevant, rewritten


def rag_check(report: str, state: dict, retriever, judge_llm) -> tuple[list[dict], dict]:
    """기술·시장 문장을 원래 문서와 대조한다 (Self-RAG 방식). (불일치 목록, 요약 정보) 반환.

    ① 검색 ② 관련성 검사 ③ 관련 없으면 검색어 재작성 후 재검색
    ④ 관련 조각을 끝내 못 찾은 문장은 '판정 제외'(틀렸다고 하지 않음 — 오탐 방지)
    ⑤ 찾은 문장만 근거성 검사: 수치가 원문에 있는가(코드) + 원문과 어긋나지 않는가(LLM 검수)
       인터넷에서 온 칸(투자금 등)의 수치는 원문 문서에 없는 게 정상이라 건너뛴다.
    """
    claim_list = extract_rag_claims(report)
    info = {"checked": 0, "sources": [], "not_found": 0, "rewritten": 0, "grounded": 0}
    if not claim_list:
        return [], info

    claims = dict(enumerate(claim_list, 1))
    relevant, info["rewritten"] = retrieve_relevant(claims, retriever, judge_llm)
    found = {i: c for i, c in claims.items() if relevant[i]}
    info["checked"] = len(claims)
    info["not_found"] = len(claims) - len(found)
    info["sources"] = sorted({_doc_label(d) for i in found for d in relevant[i]})

    rag_pairs = _pairs_of({k: state.get(k) for k in RAG_STATE_KEYS})
    web_pairs = _pairs_of({k: state.get(k) for k in WEB_STATE_KEYS})
    mismatches = []
    for i, claim in found.items():                                              # ⑤-1 수치 대조(코드)
        chunk_pairs = {(q["value"], q["unit"]) for d in relevant[i] for q in parse_quantities(_chunk_text(d))}
        for q in parse_quantities(claim):
            if not _needs_check(q) or q["unit"] == SCORE_UNIT:
                continue
            from_web = _found_in(q, web_pairs) and not _found_in(q, rag_pairs)
            if from_web or _found_in(q, chunk_pairs):
                continue
            where = ", ".join(sorted({_doc_label(d) for d in relevant[i]}))
            mismatches.append({"kind": "rag_number", "value": q["surface"], "surface": q["surface"],
                               "context": f"원문({where})의 수치와 다름 · {claim[:40]}"})

    if judge_llm is not None and found:                                        # ⑤-2 근거성 검사(LLM 검수)
        logger.info("[원문 대조] LLM 검수 호출 (원문을 찾은 문장 %d건)", len(found))
        blocks = [f"[주장 {i}] {c}\n[원문 조각]\n" + "\n".join(_chunk_text(d) for d in relevant[i])
                  for i, c in found.items()]
        for item in _ask_json(judge_llm, RAG_JUDGE_SYSTEM, blocks):
            try:
                i, verdict = int(item["id"]), item["verdict"]
            except (KeyError, TypeError, ValueError):
                continue
            if i not in found:
                continue
            if verdict == "unsupported":
                mismatches.append({"kind": "rag_claim", "value": found[i][:60],
                                   "context": "원문과 다름 · " + str(item.get("reason", ""))})
            elif verdict == "supported":
                info["grounded"] += 1
    logger.info("[원문 대조] 문장 %d건 중 원문 찾음 %d건(재검색 %d건), 불일치 %d건",
                len(claims), len(found), info["rewritten"], len(mismatches))
    return mismatches, info


# ═════════════════════════════════════════════════════════════
# 끝내 안 맞을 때의 처리
# ═════════════════════════════════════════════════════════════
def mark_unverified(report: str, mismatches: list[dict]) -> str:
    """끝내 안 맞는 표현 옆에 '(확인 필요)'를 붙인다. (설계 문서: 재작성 후에도 불일치 시)

    REFERENCE 앞 본문에서만 표시한다(출처 URL의 숫자를 건드리지 않으려는 것).
    서술 주장(claim)은 오탐 가능성이 있어 본문을 건드리지 않고 한계점에만 적는다.
    """
    head, sep, tail = report.partition(REFERENCE_HEADING)
    for m in mismatches:
        if m["kind"] not in MARKABLE_KINDS or not m.get("surface"):
            continue
        if m["kind"] == "citation":  # 존재하지 않는 인용 번호는 표시로 바꿔 넣는다
            head = head.replace(m["surface"], MARK_BAD_CITATION)
            continue
        # 같은 표현의 첫 등장 한 곳에만 표시. 앞뒤가 숫자인 경우(1200 속의 120)는 건너뛴다
        pattern = re.compile(r"(?<![\d.,])" + re.escape(m["surface"])
                             + r"(?![\d])(?!" + re.escape(MARK_UNVERIFIED) + ")")
        head = pattern.sub(lambda x: x.group(0) + MARK_UNVERIFIED, head, count=1)
    return head + sep + tail


# ═════════════════════════════════════════════════════════════
# 한계점 자동 반영 — 검증한 범위만큼만 정직하게 쓴다
# ═════════════════════════════════════════════════════════════
def build_verification_note(result: dict, scores: dict) -> str:
    """검증 결과를 사람이 읽을 문장으로 만든다. (보고서 한계점 뒤에 들어갈 내용)"""
    lines = [NOTE_HEADING]
    rule_problems = [m for m in result["mismatches"] if not m["kind"].startswith(("claim", "rag"))]
    claim_problems = [m for m in result["mismatches"] if m["kind"] == "claim"]
    rag_problems = [m for m in result["mismatches"] if m["kind"].startswith("rag")]

    if not rule_problems:
        lines.append(f"- 수치 검증: 본문 수치 {result['checked']}개를 값과 단위 기준으로 분석 자료와 대조한 결과 "
                     f"불일치가 없었습니다.")
    else:
        # 인용 번호("[9]")는 진짜 인용처럼 보이지 않게 '출처 번호 9'로 풀어 쓴다
        shown = {m["value"] if not m["value"].startswith("[") else f"출처 번호 {m['value'][1:-1]}"
                 for m in rule_problems}
        lines.append(f"- 수치·표기 검증: 재작성 후에도 분석 자료에서 확인되지 않은 항목이 있어 "
                     f"'{MARK_UNVERIFIED}'로 표시했습니다. ({', '.join(sorted(shown))})")

    rag = result.get("rag") or {}
    if rag.get("checked"):
        found = rag["checked"] - rag.get("not_found", 0)
        docs = ", ".join(rag.get("sources", [])[:4]) or "원문"
        head = (f"- 원문 대조(RAG): 기술·시장 문장 {rag['checked']}건 중 관련 원문을 찾은 {found}건을 "
                f"원문({docs})과 대조")
        if rag.get("rewritten"):
            head += f"(검색어 재작성 {rag['rewritten']}건 포함)"
        if not rag_problems:
            lines.append(head + "한 결과 불일치가 없었습니다.")
        else:
            lines.append(head + f"한 결과 {len(rag_problems)}건이 원문과 맞지 않습니다.")
            for m in rag_problems[:MAX_NOTED_CLAIMS]:
                lines.append(f"  - \"{m['value']}\" — {m['context']}")
        if rag.get("not_found"):
            lines.append(f"  - 관련 원문을 찾지 못한 {rag['not_found']}건은 원문 대조를 하지 못했습니다(분석 자료와만 대조).")
    else:
        lines.append("- 원문 대조(RAG): 벡터 DB에 연결하지 못했거나 대조할 기술·시장 문장이 없어 실행하지 않았습니다. "
                     "기술·시장 수치는 분석 자료와만 대조했습니다.")

    if result.get("judged"):
        if not claim_problems:
            lines.append(f"- 주장 검증(LLM 검수): 인용이 달린 핵심 주장 {result['judged']}건에서 "
                         f"자료와 어긋나는 내용을 찾지 못했습니다.")
        else:
            lines.append(f"- 주장 검증(LLM 검수): {result['judged']}건 중 {len(claim_problems)}건은 자료에서 "
                         f"뒷받침을 확인하지 못했습니다.")
            for m in claim_problems[:MAX_NOTED_CLAIMS]:
                lines.append(f"  - \"{m['value']}\" — {m['context']}")
    else:
        lines.append("- 주장 검증: LLM 검수를 실행하지 않아 숫자가 없는 서술 주장은 검증하지 않았습니다.")

    insufficient = find_insufficient_items(scores)
    if insufficient:
        lines.append(f"- 정보 부족 항목: {', '.join(insufficient)} "
                     f"(공개 정보가 부족해 최대 {INSUFFICIENT_MAX_SCORE}점으로 제한)")
    lines.append("- 검증 범위: 인터넷 출처 값(투자·팀·경쟁사)은 분석 에이전트가 수집한 값과의 일치만 확인했습니다. "
                 "원문 대조는 검색된 문서 조각 범위 안에서만 유효합니다.")
    return "\n".join(lines)


def add_verification_note(report: str, note: str) -> str:
    """검증 결과를 REFERENCE 바로 앞(= 한계점 장의 끝)에 넣는다. 이미 있으면 다시 넣지 않는다."""
    if NOTE_HEADING in report:
        return report
    head, sep, tail = report.partition(REFERENCE_HEADING)
    if not sep:  # REFERENCE가 없으면 끝에 붙인다
        return report.rstrip() + "\n\n" + note + "\n"
    return head.rstrip() + "\n\n" + note + "\n\n" + sep + tail


# ═════════════════════════════════════════════════════════════
# LangGraph 노드 본체
# ═════════════════════════════════════════════════════════════
def verifier_node(state: dict, judge_llm=None, retriever=None) -> dict:
    """사실 검증 노드. 통과 여부와 불일치 목록을 verify_result에 기록한다.

    judge_llm: None=자동(키가 있으면 사용), False=Judge 끄기, 객체=테스트용 가짜 LLM.
    retriever: None=자동(벡터 DB가 있으면 연결), False=원문 대조 끄기, 객체=팀원 검색기·테스트용 가짜 검색기.
    그래프에서 팀원 검색기를 쓰려면: functools.partial(verifier_node, retriever=팀원검색기)
    """
    logger.info("[사실 검증] 시작")
    report = state.get("report")
    if not report:
        raise ValueError("State에 'report'가 없습니다. 보고서 생성 노드가 먼저 실행돼야 합니다.")
    scores = state.get("scores") or {}

    # 1단계: 규칙 검증
    evidence = build_evidence(state)
    logger.info("[사실 검증] 정답지 수량 %d개 확보", len(evidence["pairs"]))
    mismatches, checked = find_quantity_mismatches(report, evidence)
    mismatches += find_wording_problems(report, state)
    mismatches += check_citations(report, collect_sources(state) if _is_recommend_mode(state) else [])
    mismatches += check_total_score(scores)

    judge = _resolve_judge(judge_llm)

    # 2단계: RAG 원문 대조 (문서에서 온 기술·시장 내용)
    rag_info = {"checked": 0, "sources": [], "not_found": 0, "rewritten": 0, "grounded": 0}
    company_name = (state.get("company") or {}).get("name")
    search = _resolve_retriever(retriever, company_name)
    if search is not None:
        rag_problems, rag_info = rag_check(report, state, search, judge)
        mismatches += rag_problems

    # 3단계: LLM 검수 (나머지 서술 주장. 원문 대조가 맡은 장은 빼서 중복 판정하지 않는다)
    judged = 0
    if judge is not None:
        claim_problems, judged = judge_claims(report, state, judge, skip_rag_sections=bool(rag_info["checked"]))
        mismatches += claim_problems

    passed = not mismatches
    logger.info("[사실 검증] 수치 %d개 검사, 원문 대조 %d건, 주장 %d건 판정, 불일치 %d개 → %s",
                checked, rag_info["checked"], judged, len(mismatches), "통과" if passed else "불일치")

    result = {"passed": passed, "mismatches": mismatches, "checked": checked, "judged": judged,
              "rag": rag_info}
    update = {"verify_result": result}

    exhausted = not passed and state.get("retry_count", 0) >= MAX_RETRY
    if passed or exhausted:
        # 검증이 끝나는 시점: 안 맞는 표현은 표시하고, 결과를 한계점에 자동으로 적는다
        final_report = report
        if exhausted:
            logger.warning("[사실 검증] 재작성 후에도 불일치 → '확인 필요' 표시 후 종료")
            final_report = mark_unverified(final_report, mismatches)
            result["finalized"] = True
        update["report"] = add_verification_note(final_report, build_verification_note(result, scores))
    return update
