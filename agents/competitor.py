"""경쟁사 비교 에이전트 (담당: 이채목)

입력: state["company"] (name, segment, product)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"competitor_analysis": {...}, "sources": [출처 dict, ...]}
  - competitor_analysis: 같은 세부 분야 경쟁사 목록, 차별점, 경쟁 리스크
    경쟁사마다 tier를 붙인다: "선도 기업"(대기업·상장사, 시장 진입 위협) / "동급 기업"(비상장 스타트업, 상대적 위치 비교)
"""
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.explorer import SEGMENT_RULES
from agents.web import format_results, search_many, to_source
from config import LLM_ATTEMPTS, LLM_MAX_TOKENS, LLM_SEED, MODEL_ANALYZE

# 세부 분야별 검색 키워드 (경쟁사를 같은 분야에서 찾기 위함)
SEGMENT_KEYWORDS = {
    "데이터센터": "데이터센터 AI 추론 가속기",
    "엣지": "온디바이스 엣지 AI NPU",
    "차량용": "차량용 자율주행 AI 반도체",
    "인프라": "CXL 메모리 인터커넥트 반도체",
    "기타": "AI 반도체",
}
# 해외 선도 기업은 국내 뉴스에 잘 잡히지 않아 영어 검색어를 함께 쓴다
SEGMENT_KEYWORDS_EN = {
    "데이터센터": "data center AI inference accelerator chip companies",
    "엣지": "edge AI NPU chip companies",
    "차량용": "automotive AI chip ADAS SoC companies",
    "인프라": "CXL switch memory interconnect chip companies",
    "기타": "AI chip companies",
}


class Competitor(BaseModel):
    name: str
    country: str = Field(description="국가. 예: 한국, 미국")
    product: str = Field(description="비교 대상 제품. 없으면 빈 문자열")
    comparison: str = Field(description="대상 기업과 비교한 한 문장 (검색 결과에 근거)")


Tier = Literal["선도 기업", "동급 기업"]


class SegmentLabel(BaseModel):
    index: int = Field(description="기업 목록의 번호")
    in_segment: bool = Field(description="이 기업이 해당 세부 분야용 칩 제품을 판매하거나 개발 중인가 (검색 결과 근거)")
    tier: Tier = Field(description="선도 기업: 대기업(계열사 포함) 또는 코스피·코스닥·해외 증시 상장사 / 동급 기업: 비상장 스타트업")
    basis: str = Field(description="tier 근거 한 구절 (검색 결과 기준). 예: 코스닥 상장사, 삼성 계열 대기업, 비상장·시리즈B")


class SegmentLabels(BaseModel):
    labels: list[SegmentLabel]


LABEL_PROMPT = """아래 기업들이 '{segment}' 분야용 칩 제품을 판매하거나 개발 중인지, 그리고 기업 구분(tier)을 검색 결과를 근거로 판정하라.
{segment_rules}
여러 분야를 하는 대기업은 '{segment}' 분야 제품이 있으면 in_segment=true다. 근거가 없으면 false로 둔다.
기업 구분: 대기업(계열사 포함)이거나 코스피·코스닥·해외 증시에 상장한 기업이면 "선도 기업", 비상장 스타트업이면 "동급 기업".
스타트업으로 출발했더라도 이미 상장을 마쳤으면 "선도 기업"이다. 반대로 상장 추진, 상장 예비심사 청구, 프리IPO 투자,
IPO 준비 단계는 아직 비상장이므로 "동급 기업"이다. 주가·시가총액·상장일이 기사에 나오면 상장사로 본다.
기업 목록 (번호 그대로 index에 넣어라):
{names}

{results}"""


class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="같은 세부 분야의 국내외 경쟁사 3~6곳. 시장을 주도하는 대기업·상장사와 비슷한 단계의 스타트업을 모두 포함")
    differentiation: list[str] = Field(description="대상 기업의 차별점. 성능·전력 효율 수치, 특허, 소프트웨어 개발 환경(SDK, 컴파일러) 등 근거가 있는 것만")
    competitive_risks: list[str] = Field(description="경쟁 관점의 리스크")
    evidence_ids: list[int] = Field(description="근거로 사용한 검색 결과 번호 [n]")


PROMPT = """너는 AI 반도체 투자 심사역이다. '{name}'({segment}, 주력 제품: {product})의 경쟁 구도를 분석하라.
경쟁사는 같은 세부 분야({segment})에서 비슷한 칩을 만드는 기업으로 고른다. 다른 분야 기업과 비교하지 않는다.
시장을 주도하는 대기업·상장사(진입 위협)와 비슷한 단계의 비상장 스타트업(상대적 위치)을 모두 찾아라.
아래 검색 결과에 근거한 내용만 쓰고, 근거로 쓴 결과 번호를 evidence_ids에 넣어라.

{results}"""


TIER_NOTE = """

[확정된 경쟁사와 기업 구분] (상장 여부 검색으로 확인한 값)
{tiers}
경쟁사는 위 기업만 이 순서대로 쓴다. 비교 문장, 차별점, 경쟁 리스크에 기업 구분을 그대로 반영하라.
선도 기업은 대기업·상장사이므로 스타트업, 신생 기업, 비슷한 단계로 쓰지 않는다.
동급 기업은 비상장 스타트업이므로 대기업이나 상장사로 쓰지 않는다."""


def _rewrite_with_tier(prompt: str, analysis: dict, bases: list[str]) -> tuple[dict, list[int]]:
    """처음 분석은 tier 판정 전에 쓰여 구분과 어긋날 수 있다 (상장사 파두를 "유사 스타트업"으로 쓴 적이 있음).
    확정된 구분을 알려주고 같은 검색 결과로 비교 문장·차별점·리스크를 다시 쓴다."""
    comps = analysis["competitors"]
    tiers = "\n".join(f"- {x['name']}: {x['tier']} ({basis})" for x, basis in zip(comps, bases))
    a = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(CompetitorAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(
        prompt + TIER_NOTE.format(tiers=tiers))
    # 다시 쓸 때 기업 순서가 바뀐 적이 있어 번호가 아니라 이름으로 짝짓는다. 짝이 없는 기업은 처음 문장을 둔다
    rewritten = {_norm(new.name): new.comparison for new in a.competitors}
    for x in comps:
        key = _norm(x["name"])
        match = rewritten.get(key) or next((t for k, t in rewritten.items() if k in key or key in k), None)
        if match:
            x["comparison"] = match
    analysis["differentiation"] = a.differentiation
    analysis["competitive_risks"] = a.competitive_risks
    return analysis, a.evidence_ids


def _norm(s: str) -> str:
    return s.split("(")[0].replace(" ", "").lower()


def competitor_node(state: dict) -> dict:
    c = state["company"]
    name, segment, product = c["name"], c.get("segment", ""), c.get("product", "")
    kw = SEGMENT_KEYWORDS.get(segment, "AI 반도체")
    results = search_many([
        f"{name} 경쟁사 비교",
        f"{kw} 시장 주요 업체",       # 선도 기업 (국내 기사)
        f"{kw} 스타트업 투자 유치",   # 동급 기업
    ], max_results=5)
    results += search_many([SEGMENT_KEYWORDS_EN.get(segment, "AI chip companies")], max_results=5, topic="general")
    prompt = PROMPT.format(name=name, segment=segment, product=product, results=format_results(results))
    a = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(CompetitorAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(prompt)
    evidence_ids = list(a.evidence_ids)
    analysis = a.model_dump(exclude={"evidence_ids"})
    # 다른 세부 분야 기업은 비교 대상에서 뺀다 (설계 1.3: 경쟁사 비교는 세부 분야 기준).
    # 분야 판정은 대상 기업 정보를 주지 않은 별도 호출로 해, "같은 분야에서 고르라"는 지시에 맞춰
    # 분류가 기우는 것을 막는다.
    # 대상 기업 자신은 경쟁사에서 뺀다 ("파네시아 (Panmnesia)" 같은 표기도 걸러지도록 포함 관계로 비교)
    analysis["competitors"] = [x for x in analysis["competitors"] if _norm(name) not in _norm(x["name"])]
    names = [x["name"] for x in analysis["competitors"]]
    if names:
        listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
        # 경쟁사마다 주력 칩과 상장 여부를 검색해 분야·기업 구분 판정의 근거로 쓴다
        # (칩 검색만으로는 상장 정보가 안 잡혀 코스닥 상장사 파두를 비상장으로 판정한 적이 있음)
        evidence = []
        for n in names:
            short = n.split("(")[0].strip()
            # 상장사는 주가·시가총액 기사가 있고 비상장사는 없다 ("상장 IPO"로 찾으면 상장 추진 기사가 섞였음)
            evidence += search_many([f"{short} {kw} 칩", f"{short} 주가 시가총액"], max_results=2, topic="general")
        labels = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(SegmentLabels).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(
            LABEL_PROMPT.format(segment=segment, segment_rules=SEGMENT_RULES, names=listing, results=format_results(evidence, limit=600)))
        label_of = {l.index: l for l in labels.labels}  # 이름 표기가 달라도 맞도록 번호로 매칭
        for i, x in enumerate(analysis["competitors"]):
            label = label_of.get(i)
            x["in_segment"] = bool(label and label.in_segment)
            x["tier"] = label.tier if label else "선도 기업"
            x["basis"] = label.basis if label else ""
        analysis["excluded_other_segment"] = [x["name"] for x in analysis["competitors"] if not x["in_segment"]]
        analysis["competitors"] = [x for x in analysis["competitors"] if x.pop("in_segment")]
        bases = [x.pop("basis") for x in analysis["competitors"]]
        if analysis["competitors"]:
            analysis, more_ids = _rewrite_with_tier(prompt, analysis, bases)
            evidence_ids += more_ids
    if not analysis["competitors"]:
        analysis["competitive_risks"].append("같은 세부 분야 경쟁사를 공개 자료로 확인하지 못함")
    used = [results[i] for i in dict.fromkeys(evidence_ids) if 0 <= i < len(results)]
    return {
        "competitor_analysis": analysis,
        "sources": [to_source(r, name, "competitor") for r in used],
    }
