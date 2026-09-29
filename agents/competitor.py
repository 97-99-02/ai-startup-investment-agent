"""경쟁사 비교 에이전트 (담당: 이채목)

입력: state["company"] (특히 segment)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"competitor_analysis": {...}, "sources": [출처 dict, ...]}
  - competitor_analysis: 같은 세부 분야 경쟁사 목록, 차별점, 경쟁 리스크
"""


def competitor_node(state: dict) -> dict:
    # TODO(채목)
    return {"competitor_analysis": {"todo": True}, "sources": []}
