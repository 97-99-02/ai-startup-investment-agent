"""스타트업 탐색 에이전트 (담당: 이채목)

도구: 웹 검색 (Tavily) / LLM: config.MODEL_EXTRACT
1) 첫 호출: config.SEARCH_QUERY로 검색해 후보를 발굴하고 조건(상장 여부, 투자 단계, 국내, AI 반도체 설계)을
   1차 판정해 최대 config.MAX_CANDIDATES개를 candidates에 저장. 검색 결과에 실제로 나온 기업명만 인정
2) 매 호출: candidates[current_index] 기업을 다시 검색해 최신 조건을 상세 확인하고 세부 분야 분류
출력: candidates, current_index(+1), company, is_eligible, eligibility_reason, sources, rejected(미충족 시)
  - 이전 기업 분석 결과는 state.RESET_ON_NEXT 로 비운다
"""
from state import RESET_ON_NEXT


def discover_candidates() -> list[str]:
    # TODO(채목)
    return ["후보A", "후보B"]


def verify_company(name: str) -> dict:
    # TODO(채목)
    return {
        "company": {"name": name, "segment": "", "round": "", "ceo": ""},
        "is_eligible": True,
        "eligibility_reason": "",
        "sources": [],
    }


def explorer_node(state: dict) -> dict:
    candidates = state.get("candidates") or discover_candidates()
    idx = state.get("current_index", 0)
    result = verify_company(candidates[idx])
    out = {**RESET_ON_NEXT, "candidates": candidates, "current_index": idx + 1, **result}
    if not result["is_eligible"]:
        out["rejected"] = [{"company": candidates[idx], "reason": result["eligibility_reason"], "total": None}]
    return out
