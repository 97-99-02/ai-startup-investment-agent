"""팀 분석 에이전트 (담당: 이채목)

입력: state["company"] (name, ceo)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"team_analysis": {...}, "sources": [출처 dict, ...]}
  - team_analysis: 창업자·핵심 인력 경력, 전문성
"""


def team_node(state: dict) -> dict:
    # TODO(채목)
    return {"team_analysis": {"todo": True}, "sources": []}
