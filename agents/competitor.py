"""경쟁사 비교 에이전트 (담당: 이채목)

입력: state["company"] (name, segment, product)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"competitor_analysis": {...}, "sources": [출처 dict, ...]}
  - competitor_analysis: 같은 세부 분야 경쟁사 목록, 차별점, 경쟁 리스크
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.explorer import SEGMENT_RULES, Segment
from agents.web import format_results, search_many, to_source
from config import MODEL_ANALYZE

# 세부 분야별 검색 키워드 (경쟁사를 같은 분야에서 찾기 위함)
SEGMENT_KEYWORDS = {
    "데이터센터": "데이터센터 AI 추론 가속기",
    "엣지": "온디바이스 엣지 AI NPU",
    "차량용": "차량용 자율주행 AI 반도체",
    "인프라": "CXL 메모리 인터커넥트 반도체",
    "기타": "AI 반도체",
}


class Competitor(BaseModel):
    name: str
    country: str = Field(description="국가. 예: 한국, 미국")
    product: str = Field(description="비교 대상 제품. 없으면 빈 문자열")
    comparison: str = Field(description="대상 기업과 비교한 한 문장 (검색 결과에 근거)")


class SegmentLabel(BaseModel):
    name: str
    segment: Segment


class SegmentLabels(BaseModel):
    labels: list[SegmentLabel]


LABEL_PROMPT = """아래 기업들의 주력 칩 세부 분야를 검색 결과를 근거로 판정하라.
{segment_rules}
기업: {names}

{results}"""


class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="같은 세부 분야의 국내외 경쟁사 3~5곳")
    differentiation: list[str] = Field(description="대상 기업의 차별점. 성능·전력 효율 수치, 특허, 소프트웨어 개발 환경(SDK, 컴파일러) 등 근거가 있는 것만")
    competitive_risks: list[str] = Field(description="경쟁 관점의 리스크")
    evidence_ids: list[int] = Field(description="근거로 사용한 검색 결과 번호 [n]")


PROMPT = """너는 AI 반도체 투자 심사역이다. '{name}'({segment}, 주력 제품: {product})의 경쟁 구도를 분석하라.
경쟁사는 같은 세부 분야({segment})에서 비슷한 칩을 만드는 기업으로 고른다. 다른 분야 기업과 비교하지 않는다.
아래 검색 결과에 근거한 내용만 쓰고, 근거로 쓴 결과 번호를 evidence_ids에 넣어라.

{results}"""


def competitor_node(state: dict) -> dict:
    c = state["company"]
    name, segment, product = c["name"], c.get("segment", ""), c.get("product", "")
    kw = SEGMENT_KEYWORDS.get(segment, "AI 반도체")
    results = search_many([f"{name} 경쟁사 비교", f"{kw} 경쟁 기업", f"{kw} 시장 주요 업체"], max_results=5)
    llm = ChatOpenAI(model=MODEL_ANALYZE, temperature=0).with_structured_output(CompetitorAnalysis)
    a = llm.invoke(PROMPT.format(name=name, segment=segment, product=product, results=format_results(results)))
    used = [results[i] for i in a.evidence_ids if 0 <= i < len(results)]
    analysis = a.model_dump(exclude={"evidence_ids"})
    # 다른 세부 분야 기업은 비교 대상에서 뺀다 (설계 1.3: 경쟁사 비교는 세부 분야 기준).
    # 분야 판정은 대상 기업 정보를 주지 않은 별도 호출로 해, "같은 분야에서 고르라"는 지시에 맞춰
    # 분류가 기우는 것을 막는다.
    names = [x["name"] for x in analysis["competitors"]]
    if names:
        labels = ChatOpenAI(model=MODEL_ANALYZE, temperature=0).with_structured_output(SegmentLabels).invoke(
            LABEL_PROMPT.format(segment_rules=SEGMENT_RULES, names=", ".join(names), results=format_results(results)))
        seg_of = {l.name: l.segment for l in labels.labels}
        for x in analysis["competitors"]:
            x["segment"] = seg_of.get(x["name"], "기타")
        analysis["excluded_other_segment"] = [x["name"] for x in analysis["competitors"] if x["segment"] != segment]
        analysis["competitors"] = [x for x in analysis["competitors"] if x["segment"] == segment]
    return {
        "competitor_analysis": analysis,
        "sources": [to_source(r, name, "competitor") for r in used],
    }
