"""설계 문서 4.2 State 설계를 그대로 옮긴 것."""
import operator
from typing import Annotated, Optional, TypedDict


class Source(TypedDict, total=False):
    """sources 리스트의 항목 형식. 모든 에이전트가 같은 형식으로 기록한다."""
    company: str        # 어느 기업 분석에 쓴 출처인지 (REFERENCE는 투자 기업 것만 사용)
    node: str           # 기록한 에이전트 이름 (explorer, tech_summary, market, competitor, team)
    kind: str           # "web" | "report" (RAG 문서)
    title: str          # 기사 제목 또는 보고서명
    publisher: str      # 언론사·발행기관
    date: str           # YYYY-MM-DD 또는 YYYY
    url: str
    snippet: str        # 근거로 쓴 원문 일부 (사실 검증에 사용)
    page: int           # RAG 문서 출처의 쪽 번호 (웹 출처는 없음)
    source_id: str      # 기술 요약 주장과 이 출처를 연결하는 키 (청크 재적재 시 변경 가능)
    chunk_id: int       # Chroma 청크 ID (RAG 문서, 재적재 시 변경 가능)
    source_file: str    # 청크가 나온 원본 문서 파일명
    source_path: str    # 프로젝트 루트 기준 원본 파일 위치 (있을 때)


class GraphState(TypedDict, total=False):
    candidates: list[str]                 # 탐색 에이전트가 발굴한 후보 (최대 5개)
    current_index: int                    # 다음에 평가할 후보 순번
    company: Optional[dict]               # 기업명, 세부 분야, 투자 단계, 대표자, 투자 금액·기업가치
    is_eligible: bool                     # 상세 확인 결과 평가 조건 충족 여부
    eligibility_reason: str

    # 병렬 분석 노드 4개는 각자 다른 키에만 기록한다
    tech_summary: Optional[dict]
    market_analysis: Optional[dict]
    competitor_analysis: Optional[dict]
    team_analysis: Optional[dict]

    # 여러 노드가 계속 추가하므로 누적(reducer)
    sources: Annotated[list[Source], operator.add]
    rejected: Annotated[list[dict], operator.add]  # {company, reason, total}

    scores: Optional[dict]   # 항목별 점수·근거·출처, legal_risk, total, decision, risks
    report: str
    verify_result: dict      # {passed, mismatches}
    retry_count: int


# 다음 후보로 넘어갈 때 비우는 키 (이전 기업 내용이 섞이지 않게)
RESET_ON_NEXT = {
    "company": None,
    "is_eligible": False,
    "eligibility_reason": "",
    "tech_summary": None,
    "market_analysis": None,
    "competitor_analysis": None,
    "team_analysis": None,
    "scores": None,
}
