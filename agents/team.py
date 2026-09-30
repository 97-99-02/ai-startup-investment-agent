"""팀 분석 에이전트 (담당: 이채목)

입력: state["company"] (name, ceo)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"team_analysis": {...}, "sources": [출처 dict, ...]}
  - team_analysis: 창업자·핵심 인력 경력, 전문성 (평가표 창업자·팀 항목의 근거)
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.web import format_results, search_many, to_source
from config import LLM_ATTEMPTS, LLM_MAX_TOKENS, LLM_SEED, MODEL_ANALYZE


class Member(BaseModel):
    name: str
    role: str = Field(description="예: 대표이사, CTO, 공동창업자")
    background: str = Field(description="학력·경력 요약 (검색 결과에 나온 것만)")


class TeamAnalysis(BaseModel):
    members: list[Member] = Field(description="창업자와 핵심 인력")
    team_size: str = Field(description="임직원 수. 없으면 빈 문자열")
    strengths: list[str] = Field(description="반도체 분야 전문성, 창업·매각 경험 등 강점")
    concerns: list[str] = Field(description="핵심 인력 이탈, 법률 이슈, 정보 부족 등 우려 사항")
    evidence_ids: list[int] = Field(description="근거로 사용한 검색 결과 번호 [n]")


PROMPT = """너는 AI 반도체 투자 심사역이다. '{name}'의 창업자와 핵심 인력을 분석하라.
대표자로 알려진 인물: {ceo}
아래 검색 결과에 근거한 내용만 쓰고, 확인되지 않은 경력은 쓰지 마라. 근거로 쓴 결과 번호를 evidence_ids에 넣어라.
정보가 부족하면 concerns에 "공개 정보 부족"을 적는다.

{results}"""


def team_node(state: dict) -> dict:
    c = state["company"]
    name, ceo = c["name"], c.get("ceo", "")
    queries = [f"{name} 창업자 CTO 경력", f"{name} 대표 인터뷰"]
    if ceo:
        queries.insert(0, f"{name} {ceo} 대표 경력")
    results = search_many(queries, max_results=5, topic="general")
    results = [r for r in results if name in r["title"] + r["content"]]
    llm = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(TeamAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    a = llm.invoke(PROMPT.format(name=name, ceo=ceo or "확인되지 않음", results=format_results(results)))
    used = [results[i] for i in a.evidence_ids if 0 <= i < len(results)]
    return {
        "team_analysis": a.model_dump(exclude={"evidence_ids"}),
        "sources": [to_source(r, name, "team") for r in used],
    }
