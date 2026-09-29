# AI Semiconductor Startup Investment Evaluation Agent

본 프로젝트는 국내 AI 반도체 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview
- Objective : AI 반도체 스타트업의 기술·제품 성숙도, 시장성, 창업자·팀, 실적, 경쟁 우위, 투자조건을 기준으로 투자 적합성 분석
- Method : LangGraph Multi-Agent, Agentic RAG
- Tools : Tavily 웹 검색, Chroma 벡터 DB

## Features
- 웹 검색 기반 평가 후보 발굴과 조건(비상장, Seed ~ Series C, Exit 전) 필터링
- 공공 연구기관 보고서(10편, 182쪽) 기반 기술·시장 분석 (RAG)
- 평가표 기반 채점과 규칙 기반 투자/보류 결정, 보류 시 다음 후보로 반복
- 출처가 붙은 투자 보고서 생성과 수치 사실 검증

## Tech Stack
- Framework : LangGraph
- LLM/Generator : gpt-4.1-mini (탐색은 gpt-4.1-nano)
- LLM/Judge : gpt-4.1
- Retrieval : Chroma - Hit Rate@K {TODO}, MRR {TODO}
- Embedding : {TODO: 비교 실험 후 확정 (후보 bge-m3 / KURE-v1 / Qwen3-Embedding-0.6B)}

## Agents
- 스타트업 탐색 : 웹 검색으로 후보 발굴, 조건 확인, 세부 분야 분류
- 기술 요약 : RAG(기술 문서)로 칩·공정·개발 단계·강점·약점 요약
- 시장성 평가 : RAG(시장 보고서)로 세부 분야 시장 규모·성장률 분석
- 경쟁사 비교 : 웹 검색으로 경쟁사 목록과 차별점, 경쟁 리스크 정리
- 팀 분석 : 웹 검색으로 창업자·핵심 인력 역량 분석
- 투자 판단 : 평가표 채점(LLM) 후 환산 점수·투자 결정(규칙)
- 보고서 생성 / 사실 검증 : 보고서 작성, 수치와 출처 대조

## Architecture
(그래프 이미지 TODO)

## Directory Structure
```
├── data/            # RAG 문서 풀 (tech/, market/)
├── agents/          # 에이전트 모듈 (에이전트별 파일)
├── rag/             # 문서 적재(ingest), 임베딩, 검색기
├── prompts/         # 프롬프트 템플릿
├── eval/            # 임베딩 비교·검색 성능 평가
├── outputs/         # 생성된 보고서
├── state.py         # 공용 State 정의
├── graph.py         # LangGraph 흐름
├── config.py        # 모델·가중치·임계값 설정
├── app.py           # 실행 스크립트
└── README.md
```

## Usage
```bash
uv sync                           # uv가 없으면: pip install -r requirements.txt
cp .env.example .env              # 키 입력
uv run python -m rag.ingest       # 최초 1회: 문서 임베딩 → vectorstore/
uv run python app.py
```

## Contributors
- 강지수 : 시장성 평가 에이전트
- 김다은 : 투자 판단 에이전트
- 이진우 : 보고서 생성, 사실 검증
- 이채목 : 스타트업 탐색, 경쟁사 비교, 팀 분석 에이전트, RAG 공통 기반, 그래프 연결
- 장정훈 : 기술 요약 에이전트
