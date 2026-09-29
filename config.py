"""설계 문서의 고정값을 한곳에 모은다. 값을 바꾸면 설계 문서도 같이 고친다."""

# 2.1 LLM 배분 기준
MODEL_EXTRACT = "gpt-4.1-nano"   # 스타트업 탐색 (단순 추출·판정)
MODEL_ANALYZE = "gpt-4.1-mini"   # 기술 요약, 시장성, 경쟁사, 팀, 보고서
MODEL_JUDGE = "gpt-4.1"          # 투자 판단 (생성 모델과 다른 모델)

# 2.3 임베딩 (비교 실험 후 확정)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# 2.4 Vector DB
CHROMA_DIR = "vectorstore"
DOC_TYPES = ("tech", "market")   # 메타데이터 doc_type 값

# 2.1 탐색 규칙
SEARCH_QUERY = "국내 AI 반도체 스타트업 투자 유치 시리즈"
MAX_CANDIDATES = 5

# 3.2 평가 항목과 비중 (합계 100)
WEIGHTS = {
    "tech": 30,         # 기술·제품 성숙도
    "market": 25,       # 시장성
    "team": 20,         # 창업자·팀
    "traction": 10,     # 실적·고객 검증
    "competition": 10,  # 경쟁 우위
    "deal": 5,          # 투자조건
}

# 3.4 투자 결정 규칙
INVEST_THRESHOLD = 70
CORE_ITEMS = ("team", "tech")   # 2점 이하이면 보류
CORE_MIN_SCORE = 3

# 4.1 사실 검증 재작성 횟수
MAX_REPORT_RETRY = 1
