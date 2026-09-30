"""설계 문서의 고정값을 한곳에 모은다. 값을 바꾸면 설계 문서도 같이 고친다."""

# 2.1 LLM 배분 기준
MODEL_ANALYZE = "gpt-4.1-mini"   # 탐색, 기술 요약, 시장성, 경쟁사, 팀, 보고서
# (탐색에 gpt-4.1-nano를 썼으나 해외 기업을 국내로 잘못 판정하고 대표자·제품 칸을 비워 mini로 통일)
MODEL_JUDGE = "gpt-4.1"          # 투자 판단 (생성 모델과 다른 모델)
# 분석·판단 LLM의 출력 상한과 재시도. 드물게 같은 내용을 반복 생성해 32,768토큰까지 가다 실패한 적이 있어
# 상한을 두어 빨리 실패시키고 한 번 더 시도한다 (결과는 JSON이라 4,096토큰이면 충분)
LLM_MAX_TOKENS = 4096
LLM_ATTEMPTS = 2
# 같은 입력이면 최대한 같은 응답이 나오게 한다 (OpenAI seed: 완전한 보장은 아님). temperature=0만으로는
# 같은 기업 분석이 호출마다 달라져 점수가 흔들렸다
LLM_SEED = 42

# 2.3 임베딩: 후보 3종 비교 실험(eval/embedding) 결과 MRR@5·Hit@3 최고, 인코딩 최속
EMBEDDING_MODEL = "nlpai-lab/KURE-v1"

# 2.4 Vector DB
CHROMA_DIR = "vectorstore"
DOC_TYPES = ("tech", "market")   # 메타데이터 doc_type 값

# 2.1 탐색 규칙
SEARCH_QUERIES = [          # 실행마다 후보가 크게 바뀌지 않도록 고정. 세부 분야별로 한 개 이상
    "국내 AI 반도체 스타트업 시리즈 투자 유치",
    "NPU 스타트업 투자 유치 시리즈",
    "LLM 추론 전용 칩 스타트업 투자 유치",       # 데이터센터
    "온디바이스 엣지 AI 반도체 스타트업 투자",    # 엣지
    "차량용 AI 반도체 스타트업 투자 유치",        # 차량용
    "CXL 메모리 반도체 스타트업 투자 유치",       # 인프라
]
ALLOWED_ROUNDS = ("Seed", "Pre-A", "Series A", "Series B", "Series C")
MAX_CANDIDATES = 5

# 3.2 평가 항목과 비중 (합계 100)
WEIGHTS = {
    "tech": 20,         # 기술·제품 성숙도 (양산 전 기업은 최대 3점이라 30에서 하향)
    "market": 30,       # 시장성
    "team": 25,         # 창업자·팀
    "traction": 10,     # 실적·고객 검증
    "competition": 10,  # 경쟁 우위
    "deal": 5,          # 투자조건
}

# 3.4 투자 결정 규칙
INVEST_THRESHOLD = 70
CORE_ITEMS = ("team", "tech")   # 1점 이하이면 보류 (핵심 역량 미달)
CORE_MIN_SCORE = 2              # 이 점수 미만(1점 이하)이면 보류
CORE_CAUTION_SCORE = 2           # 핵심 항목이 이 점수 이하이면 보류가 아니어도 보고서에 주의 코멘트를 남긴다

# 4.1 사실 검증 재작성 횟수
MAX_REPORT_RETRY = 1
