"""보고서 생성 에이전트 (담당: 이진우)

입력: 전체 State
LLM: config.MODEL_ANALYZE - 본문 문장 작성 (설계 5.1 목차)
코드: 평가표 점수 표는 scores, REFERENCE는 sources에서 생성 (투자 결정 기업 출처만)
출력: {"report": 보고서 본문(markdown), "retry_count": 재작성 횟수}
  - 사실 검증 후 다시 호출되면(verify_result 존재) retry_count를 1 올린다
  - scores["decision"] 이 "투자"가 아니면 추천 기업 없음 보고서 (rejected 중심)
"""


def reporter_node(state: dict) -> dict:
    # TODO(진우): 구현 전까지 임시 보고서를 반환한다
    retry = state.get("retry_count", 0) + (1 if state.get("verify_result") else 0)
    return {"report": "# SUMMARY\n(보고서 생성 미구현)\n", "retry_count": retry}
