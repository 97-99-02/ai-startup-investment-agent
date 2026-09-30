"""시장성 평가 에이전트 (담당: 강지수)

입력: state["company"] (name, segment: 데이터센터/엣지/차량용/인프라/기타)
도구: RAG - 시장 문서(market)를 먼저 검색하고, 세부 분야 수치가 부족하면 기술 문서(tech)를 보조로 1회 재검색
      (시장 문서는 데이터센터 위주라 차량용·엣지 시장 전망은 기술 문서에만 있다)
LLM: config.MODEL_ANALYZE - 문서 조각에서 수치·수요 요인·리스크 추출만 한다
코드: 수치가 근거 조각에 실제로 있는지 대조, 대상 세부 분야 수치인지 판정, 충분성 판단(재검색 여부), 출처 정리
출력: {"market_analysis": {...}, "sources": [출처 dict, ...]}
  - market_analysis: segment, figures(시장 규모·성장률, 원문 표기 그대로 + 출처), demand_drivers,
    market_risks, summary, searched(검색한 문서 유형), info_insufficient(설계 3.5 정보 부족 표시)
"""
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import LLM_ATTEMPTS, LLM_MAX_TOKENS, LLM_SEED, MODEL_ANALYZE
from rag.retriever import get_retriever, to_source

# 세부 분야별 검색어. 실행마다 결과가 달라지지 않도록 고정한다 (시장 규모 / 성장률 / 수요 요인)
SEGMENT_QUERIES = {
    "데이터센터": ["데이터센터 AI 반도체 시장 규모 전망", "AI 가속기 시장 연평균 성장률", "데이터센터 AI 반도체 수요 증가 요인"],
    "엣지": ["온디바이스 AI 반도체 시장 규모 전망", "엣지 AI NPU 시장 성장률", "온디바이스 AI 수요 요인"],
    "차량용": ["자율주행 차량용 AI 반도체 시장 규모 전망", "차량용 반도체 시장 성장률", "자율주행 AI 칩 수요 요인"],
    "인프라": ["HBM AI 메모리 시장 규모 전망", "CXL 메모리 시장 성장률", "AI 메모리 수요 요인"],
    "기타": ["AI 반도체 시장 규모 전망", "AI 반도체 시장 연평균 성장률", "AI 반도체 수요 요인"],
}

# 수치가 대상 세부 분야의 것인지 코드로 판정할 때 쓰는 용어. scope와 근거 조각 원문에 모두 있어야 한다
# (LLM에게 판정을 맡기면 데이터센터 수치를 '차량용'으로 바꿔 적는 오류가 있었다)
SEGMENT_TERMS = {
    "데이터센터": ("데이터센터", "서버", "클라우드", "하이퍼스케일"),
    "엣지": ("온디바이스", "엣지", "on-device", "pc", "스마트폰", "모바일", "로봇"),
    "차량용": ("차량", "자율주행", "자동차", "모빌리티", "adas"),
    "인프라": ("hbm", "cxl", "hbf", "메모리", "인터커넥트"),
    "기타": ("ai반도체",),
}

# 프롬프트에 넣을 분야 이름. '기타'를 그대로 넣으면 차트 범례의 '기타' 항목을 대상 분야로 착각한다
SEGMENT_LABELS = {"기타": "AI 반도체 전체"}

# 값에 단위가 있어야 수치로 인정한다. 단위 없는 숫자는 대부분 차트 눈금·범례를 OCR한 값이라 대상이 불분명하다
UNIT_PATTERNS = {"시장규모": re.compile(r"달러|원|\$|usd", re.I), "성장률": re.compile(r"%")}


class MarketFigure(BaseModel):
    kind: Literal["시장규모", "성장률"] = Field(description="시장규모: 금액(달러·원). 성장률: 연평균 성장률(CAGR, %)")
    value: str = Field(description="숫자와 단위만, 원문 표기 그대로. 문장을 넣지 않는다. 예: 1,240억 달러, 38%")
    year: str = Field(description="기준 연도 또는 기간. 예: 2030, 2024~2030. 없으면 빈 문자열")
    scope: str = Field(description="이 수치가 가리키는 시장 이름. 원문에 적힌 그대로 옮긴다. 예: 데이터센터용 AI반도체, 전체 AI반도체")
    evidence_id: int = Field(description="이 수치가 나온 문서 조각 번호 [n]")


class MarketAnalysis(BaseModel):
    figures: list[MarketFigure] = Field(description="시장 규모와 성장률 수치")
    demand_drivers: list[str] = Field(description="대상 세부 분야의 수요 증가 요인 (문서에 근거가 있는 것만)")
    market_risks: list[str] = Field(description="대상 세부 분야의 시장 리스크. 예: 공급 과잉, 빅테크 자체 칩, 수출 규제")
    summary: str = Field(description="세부 분야 시장을 요약한 2~3문장. figures에 넣은 수치만 쓰고, 세부 분야 수치가 없으면 '정보 부족'을 명시")
    evidence_ids: list[int] = Field(description="근거로 사용한 문서 조각 번호 [n]")


PROMPT = """너는 AI 반도체 투자 심사역이다. '{name}'이 속한 세부 분야({segment}) 시장을 분석하라.
아래 문서 조각에 있는 수치만 쓰고, 숫자는 원문 표기 그대로 옮겨라. 계산하거나 단위를 환산하지 마라.
문서 조각에 나온 시장 규모·성장률 수치는 빠짐없이 넣는다. value에는 반드시 단위(억 달러, % 등)를 붙이고,
단위나 대상 시장이 분명하지 않은 숫자(차트 눈금·범례 값 등)는 넣지 않는다.
figures 항목 하나에는 수치 하나만 넣는다. "2024년 A에서 2030년 B로 연평균 C% 성장"이라는 문장이면
시장규모 A(2024), 시장규모 B(2030), 성장률 C(2024~2030) 세 항목으로 나눈다.
scope는 원문에 적힌 시장 이름을 그대로 쓰고, 대상 분야({segment})에 맞춰 바꾸지 마라.
대상 분야가 아닌 수치(전체 AI 반도체·다른 분야 시장 등)도 참고로 넣는다.
수요 요인과 리스크는 대상 세부 분야({segment})와 직접 관련된 것만 쓴다. 한 문장에 여러 분야 요인이 나열되어 있으면
대상 분야 것만 고른다 (예: 차량용이면 'AI PC 보급'은 넣지 않는다).
summary는 대상 세부 분야 수치로 시작하고, 다른 분야 수치는 비교가 필요할 때만 한 번 언급한다.
근거로 쓴 조각 번호를 evidence_id와 evidence_ids에 넣어라.

{chunks}"""


def retrieve(queries: list[str], doc_type: str) -> list:
    """여러 검색어 결과를 청크 기준으로 중복 제거해 합친다."""
    retriever = get_retriever(doc_type)
    seen, merged = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            key = d.metadata.get("chunk_id", (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:50]))
            if key not in seen:
                seen.add(key)
                merged.append(d)
    return merged


def format_chunks(docs: list) -> str:
    """LLM 프롬프트에 넣을 형태. [번호]로 조각을 구분해 근거를 고를 수 있게 한다."""
    return "\n\n".join(
        f"[{i}] {d.metadata.get('title', d.metadata.get('source', ''))} p.{d.metadata.get('page', '')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def _compact(text: str) -> str:
    return re.sub(r"[\s,]", "", text)


def number_in_source(value: str, text: str) -> bool:
    """값에 들어 있는 숫자가 모두 원문 조각에 있는가. LLM이 수치를 잘못 옮긴 경우를 거른다."""
    nums = re.findall(r"\d+(?:\.\d+)?", _compact(value))
    body = _compact(text)
    return bool(nums) and all(n in body for n in nums)


def has_unit(kind: str, value: str) -> bool:
    """시장규모는 통화 단위, 성장률은 %가 값에 있는가."""
    return bool(UNIT_PATTERNS[kind].search(value))


def is_segment_specific(scope: str, text: str, segment: str) -> bool:
    """scope와 근거 조각 원문에 같은 세부 분야 용어가 모두 있으면 대상 분야 수치로 본다.
    기타(AI 반도체 전체)는 scope에 다른 분야 용어가 있으면 제외한다 ('데이터센터용 AI반도체'도 'ai반도체'를 포함하므로)
    """
    scope, body = _compact(scope).lower(), _compact(text).lower()
    if segment not in SEGMENT_TERMS or segment == "기타":
        others = (t for seg, terms in SEGMENT_TERMS.items() if seg != "기타" for t in terms)
        if any(t in scope for t in others):
            return False
    return any(t in scope and t in body for t in SEGMENT_TERMS.get(segment, SEGMENT_TERMS["기타"]))


def extract(name: str, segment: str, docs: list) -> tuple[MarketAnalysis, list[dict]]:
    """LLM으로 추출한 뒤, 근거 번호가 맞고 단위가 있으며 수치가 원문에 있는 것만 남기고 세부 분야 여부를 판정한다."""
    llm = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(MarketAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    label = SEGMENT_LABELS.get(segment, segment)
    a = llm.invoke(PROMPT.format(name=name, segment=label, chunks=format_chunks(docs)))
    figures = []
    for f in a.figures:
        if not (0 <= f.evidence_id < len(docs)):
            continue
        d = docs[f.evidence_id]
        if not has_unit(f.kind, f.value) or not number_in_source(f.value, d.page_content):
            continue
        figures.append({
            **f.model_dump(),
            "segment_specific": is_segment_specific(f.scope, d.page_content, segment),
            "source": f"{d.metadata.get('title', '')} p.{d.metadata.get('page', '')}",
        })
    return a, figures


def is_sufficient(figures: list[dict]) -> bool:
    """채점 기준(설계 3.3)에 필요한 세부 분야 시장 규모와 성장률이 모두 있는가."""
    kinds = {f["kind"] for f in figures if f["segment_specific"]}
    return {"시장규모", "성장률"} <= kinds


def market_node(state: dict) -> dict:
    c = state["company"]
    name, segment = c["name"], c.get("segment") or "기타"
    queries = SEGMENT_QUERIES.get(segment, SEGMENT_QUERIES["기타"])

    docs = retrieve(queries, "market")
    a, figures = extract(name, segment, docs)
    searched = ["market"]
    if not is_sufficient(figures):
        # 세부 분야 수치가 없으면 기술 문서의 시장 전망 부분으로 1회 재검색
        docs = docs + retrieve(queries, "tech")
        a, figures = extract(name, segment, docs)
        searched.append("tech")

    # 출처는 남은 수치와 LLM이 근거로 고른 조각만 (REFERENCE에 쓰지 않은 문서가 들어가지 않게)
    used = sorted({f.pop("evidence_id") for f in figures} | {i for i in a.evidence_ids if 0 <= i < len(docs)})
    analysis = {
        "segment": segment,
        "figures": figures,
        "demand_drivers": a.demand_drivers,
        "market_risks": a.market_risks,
        "summary": a.summary or "정보 부족",
        "searched": searched,
        "info_insufficient": not is_sufficient(figures),
    }
    return {
        "market_analysis": analysis,
        "sources": [to_source(docs[i], name, "market") for i in used],
    }
