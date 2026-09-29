# RAG 문서 풀 (총 10편, 182쪽)

설계 문서 2.2 기준. 모두 공공·정책 연구기관의 무료 공개 보고서다. `rag/ingest.py`가 폴더 이름을 문서 유형(doc_type)으로, `rag/catalog.py`의 제목·발행기관·연도·URL을 메타데이터로 저장한다. 제목은 각 PDF 첫 쪽의 원문 제목이다.

## tech/ (기술 요약 에이전트, 7편 95쪽)

| 파일 | 문서 | 발행기관 (연도) | 쪽 |
|---|---|---|---|
| etri210_aichip_compiler.pdf | 최신 인공지능 반도체 및 컴파일러 지원 기술 동향 | ETRI 전자통신동향분석 39권 5호 (2024) | 11 |
| etri220_aimemory_market.pdf | AI 메모리 반도체 분야 주요 기술 및 시장 동향 | ETRI 전자통신동향분석 41권 3호 (2026) | 13 |
| etri216_chiplet_llm.pdf | 칩렛 이종집적 첨단패키지 기반 LLM 가속기 설계 동향 | ETRI 전자통신동향분석 40권 5호 (2025) | 9 |
| etri218_hpc_proc.pdf | AI 워크로드를 위한 고성능 컴퓨팅(HPC) 프로세서 기술 발전 동향 | ETRI 전자통신동향분석 41권 1호 (2026) | 13 |
| etri215_auto_aichip.pdf | AI 정의 차량(ADV) 기반의 자율주행차용 AI반도체: 기술·시장 전망 및 전략적 시사점 | ETRI 전자통신동향분석 40권 4호 (2025) | 13 |
| etri217_physicalai_chip.pdf | 피지컬 AI 시대의 지각 특화 AI 반도체 개념 및 전략적 투자 방향 | ETRI 전자통신동향분석 40권 6호 (2025) | 13 |
| kisdi_outlook21_ondevice.pdf | Physical AI 시대에 대응한 On-device AI 반도체 경쟁력 강화 방향 | 정보통신정책연구원 KISDI AI Outlook (2025) | 23 |

ETRI 보고서 원문: https://ettrends.etri.re.kr

## market/ (시장성 평가 에이전트, 3편 87쪽)

| 파일 | 문서 | 발행기관 (연도) | 쪽 |
|---|---|---|---|
| exim_dc_aisemi_2026.pdf | 데이터센터용 AI반도체 시장 현황 및 전망 | 한국수출입은행 해외경제연구소 (2026) | 53 |
| kdb_aisemi_tech_industry.pdf | AI 반도체 기술 및 산업 동향 | KDB미래전략연구소 (2024) | 25 |
| kita_semi_2026.pdf | 반도체 전방산업 업황 진단 및 2026년 전망 | 한국무역협회 (2025) | 9 |
