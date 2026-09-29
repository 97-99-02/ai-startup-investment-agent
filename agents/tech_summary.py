"""기술 요약 에이전트 (담당: 장정훈)

입력: state["company"] (name, segment 등)
도구: RAG - rag.retriever.get_retriever("tech") / 필요 시 웹 검색
LLM: config.MODEL_ANALYZE
출력: {"tech_summary": {...}, "sources": [출처 dict, ...]}
  - tech_summary: 핵심 칩, 공정, 개발 단계(테이프아웃/양산), 성능 지표, 강점, 약점, 공개 매출·계약
  - sources 항목 형식은 state.Source 참고 (company 필수)
"""


def tech_summary_node(state: dict) -> dict:
    # TODO(정훈): 구현 전까지 그래프가 돌도록 임시 값을 반환한다
    return {"tech_summary": {"todo": True}, "sources": []}
