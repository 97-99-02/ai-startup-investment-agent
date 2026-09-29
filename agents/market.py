"""시장성 평가 에이전트 (담당: 강지수)

입력: state["company"] (특히 segment: 데이터센터/엣지/차량용/인프라)
도구: RAG - rag.retriever.get_retriever("market")
LLM: config.MODEL_ANALYZE
출력: {"market_analysis": {...}, "sources": [출처 dict, ...]}
  - market_analysis: 세부 분야 시장 규모, 성장률, 수요 요인 (수치마다 출처)
"""


def market_node(state: dict) -> dict:
    # TODO(지수): 구현 전까지 그래프가 돌도록 임시 값을 반환한다
    return {"market_analysis": {"todo": True}, "sources": []}
