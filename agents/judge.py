"""투자 판단 에이전트 (담당: 김다은)

입력: company, tech_summary, market_analysis, competitor_analysis, team_analysis
LLM: config.MODEL_JUDGE - 항목별 점수(1~5)와 근거, 법률 리스크 여부만 판정 (설계 3.3)
코드: 환산 점수 계산, config.INVEST_THRESHOLD 비교, 보류 조건 적용 (설계 3.4)
출력: {"scores": {...}, "rejected": [...](보류일 때)}
  - scores 필수 키: "decision" ("투자" 또는 "보류"), "total" (환산 점수)
    graph.route_after_judge 가 scores["decision"] 으로 분기한다
  - rejected 항목: {"company": 기업명, "reason": 보류 사유, "total": 환산 점수}
"""


def judge_node(state: dict) -> dict:
    # TODO(다은): 구현 전까지 모든 후보를 보류로 처리해 반복 경로가 돌게 한다
    name = state["company"]["name"]
    return {
        "scores": {"decision": "보류", "total": 0},
        "rejected": [{"company": name, "reason": "투자 판단 미구현", "total": 0}],
    }
