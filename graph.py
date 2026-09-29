"""설계 문서 4.1 Graph 흐름."""
from langgraph.graph import END, START, StateGraph

from agents.competitor import competitor_node
from agents.explorer import explorer_node
from agents.judge import judge_node
from agents.market import market_node
from agents.reporter import reporter_node
from agents.team import team_node
from agents.tech_summary import tech_summary_node
from agents.verifier import verifier_node
from config import MAX_REPORT_RETRY
from state import GraphState

ANALYSIS_NODES = ["tech_summary", "market", "competitor", "team"]


def has_remaining(state: GraphState) -> str:
    return "explorer" if state["current_index"] < len(state["candidates"]) else "reporter"


def route_after_explore(state: GraphState):
    # 조건 분기 1: 상세 확인에서 조건 미충족이면 분석을 건너뛴다
    return ANALYSIS_NODES if state["is_eligible"] else has_remaining(state)


def route_after_judge(state: GraphState) -> str:
    # 조건 분기 2: 투자면 보고서, 보류면 남은 후보 확인
    return "reporter" if state["scores"]["decision"] == "투자" else has_remaining(state)


def route_after_verify(state: GraphState) -> str:
    # 조건 분기 3: 불일치면 1회 재작성, 그래도 불일치면 확인 필요 표시 후 종료
    if state["verify_result"]["passed"] or state.get("retry_count", 0) >= MAX_REPORT_RETRY:
        return END
    return "reporter"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("explorer", explorer_node)
    g.add_node("tech_summary", tech_summary_node)
    g.add_node("market", market_node)
    g.add_node("competitor", competitor_node)
    g.add_node("team", team_node)
    g.add_node("judge", judge_node)
    g.add_node("reporter", reporter_node)
    g.add_node("verifier", verifier_node)

    g.add_edge(START, "explorer")
    g.add_conditional_edges("explorer", route_after_explore, ANALYSIS_NODES + ["explorer", "reporter"])
    g.add_edge(ANALYSIS_NODES, "judge")  # 병렬 분석 4개가 모두 끝나야 투자 판단
    g.add_conditional_edges("judge", route_after_judge, ["reporter", "explorer"])
    g.add_edge("reporter", "verifier")
    g.add_conditional_edges("verifier", route_after_verify, ["reporter", END])
    return g.compile()
