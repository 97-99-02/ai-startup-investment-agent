"""사실 검증 노드 (담당: 이진우) - 규칙 기반, LLM 없음

입력: report, sources (snippet에 원문 근거)
출력: {"verify_result": {"passed": bool, "mismatches": [출처와 맞지 않는 수치]}}
  - 재작성 한도(config.MAX_REPORT_RETRY)를 넘겨도 불일치하면 report에 "확인 필요" 표시를 달아 함께 반환
  - graph.route_after_verify 가 verify_result["passed"] 로 분기한다
"""


def verifier_node(state: dict) -> dict:
    # TODO(진우): 구현 전까지 항상 통과시킨다
    return {"verify_result": {"passed": True, "mismatches": []}}
