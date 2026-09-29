"""스타트업 탐색 에이전트 (담당: 이채목)

도구: 웹 검색 (Tavily) / LLM: 발굴 config.MODEL_EXTRACT(nano), 상세 확인 config.MODEL_ANALYZE(mini)
1) 첫 호출: config.SEARCH_QUERIES로 검색해 후보를 발굴하고 조건(상장 여부, 투자 단계, 국내, AI 반도체 설계)을
   1차 판정해 최대 config.MAX_CANDIDATES개를 candidates에 저장. 검색 결과에 실제로 나온 기업명만 인정
2) 매 호출: candidates[current_index] 기업을 다시 검색해 최신 조건을 상세 확인하고 세부 분야 분류
출력: candidates, current_index(+1), company, is_eligible, eligibility_reason, sources, rejected(미충족 시)
  - 이전 기업 분석 결과는 state.RESET_ON_NEXT 로 비운다
  - LLM은 검색 결과에서 사실을 뽑기만 하고, 조건 충족 여부는 is_eligible_round() 등 코드로 판정한다
"""
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.web import format_results, search_many, to_source
from config import ALLOWED_ROUNDS, MAX_CANDIDATES, MODEL_ANALYZE, MODEL_EXTRACT, SEARCH_QUERIES
from state import RESET_ON_NEXT

Listed = Literal["상장", "비상장", "확인불가"]
Segment = Literal["데이터센터", "엣지", "차량용", "인프라", "기타"]


# ---------- 1차: 후보 발굴 ----------
class Candidate(BaseModel):
    name: str = Field(description="기사에 표기된 기업명 그대로")
    is_domestic: bool = Field(description="한국 기업인가")
    is_ai_chip_designer: bool = Field(description="AI 연산용 반도체(NPU, AI 가속기, AI SoC 등)를 직접 설계하는 기업인가")
    listed: Listed
    latest_round: str = Field(description="기사에 나온 가장 최근 투자 단계. 예: 시드, 프리A, 시리즈A, 시리즈B 브릿지, 시리즈C, 프리IPO, 시리즈D. 없으면 '확인불가'")
    exited: bool = Field(description="M&A 또는 IPO로 엑싯했는가")


class CandidateList(BaseModel):
    candidates: list[Candidate]


DISCOVER_PROMPT = """다음은 AI 반도체 스타트업 투자 관련 뉴스 검색 결과다.
검색 결과에 등장하는 기업 중 AI 반도체와 관련된 기업을 모두 뽑아 항목별로 판정하라.
검색 결과에 없는 정보는 추측하지 말고 '확인불가' 또는 false로 둔다.

{results}"""


def normalize_round(text: str) -> str | None:
    """기사 표기를 ALLOWED_ROUNDS 형식으로 바꾼다. 허용 범위 밖이면 None."""
    t = text.replace(" ", "").lower()
    if not t or "확인불가" in t or "ipo" in t or "상장" in t:
        return None
    if "시드" in t or "seed" in t:
        return "Seed"
    if "프리a" in t or "pre-a" in t or "프리시리즈a" in t:
        return "Pre-A"
    m = re.search(r"(?:시리즈|series)([a-z])", t)
    if m:
        label = f"Series {m.group(1).upper()}"
        return label if label in ALLOWED_ROUNDS else None
    return None


def is_eligible(listed: str, round_text: str, exited: bool) -> tuple[bool, str]:
    """과제 스타트업 조건: 비상장, Seed ~ Series C, Exit 전. 코드로 판정한다."""
    if listed == "상장":
        return False, "상장 기업"
    if exited:
        return False, "M&A 또는 IPO로 엑싯"
    rnd = normalize_round(round_text)
    if rnd is None:
        return False, f"투자 단계가 Seed ~ Series C 범위가 아니거나 확인 불가 ({round_text})"
    return True, f"비상장, {rnd}"


def discover_candidates() -> list[str]:
    results = search_many(SEARCH_QUERIES, max_results=10)
    corpus = " ".join(r["title"] + " " + r["content"] for r in results)
    llm = ChatOpenAI(model=MODEL_EXTRACT, temperature=0).with_structured_output(CandidateList)
    found = llm.invoke(DISCOVER_PROMPT.format(results=format_results(results))).candidates

    picked = []
    for c in found:
        if c.name in picked or c.name not in corpus:  # 검색 결과에 실제로 나온 기업명만 인정
            continue
        ok, _ = is_eligible(c.listed, c.latest_round, c.exited)
        if ok and c.is_domestic and c.is_ai_chip_designer:
            picked.append(c.name)
        if len(picked) >= MAX_CANDIDATES:
            break
    return picked


# ---------- 2차: 기업별 상세 확인 ----------
SEGMENT_RULES = """세부 분야(segment) 분류 기준: 주력 칩이 어디에 들어가는지로 판단한다.
- 데이터센터: 서버, 클라우드, LLM 학습·추론용 가속기
- 엣지: 카메라, 로봇, 가전, 산업용 기기 등 기기 안에서 AI를 돌리는 온디바이스 칩 (엣지 NPU, AI SoC)
- 차량용: 자율주행, ADAS, 모빌리티용 칩
- 인프라: AI 칩이 아니라 메모리, CXL, 인터커넥트처럼 AI 시스템을 받치는 반도체
- 기타: 위 어디에도 해당하지 않을 때만
"""

class CompanyProfile(BaseModel):
    is_domestic: bool = Field(description="한국 기업인가")
    segment: Segment = Field(description="주력 칩의 용도 (아래 분류 기준을 따른다)")
    product: str = Field(description="대표 칩·제품명. 없으면 빈 문자열")
    ceo: str = Field(description="대표이사 이름. 없으면 빈 문자열")
    listed: Listed
    exited: bool
    latest_round: str = Field(description="가장 최근 투자 단계 (기사 표기 그대로)")
    round_date: str = Field(description="최근 투자 시기 YYYY-MM. 없으면 빈 문자열")
    funding_amount: str = Field(description="최근 라운드 투자 금액. 예: 700억 원. 없으면 빈 문자열")
    total_funding: str = Field(description="누적 투자 금액. 없으면 빈 문자열")
    valuation: str = Field(description="기업가치. 없으면 빈 문자열")
    investors: list[str] = Field(description="최근 라운드 주요 투자사")


VERIFY_PROMPT = """다음은 '{name}'에 대한 뉴스 검색 결과다. 이 기업의 정보를 항목별로 정리하라.
여러 기사가 다르면 가장 최근 기사를 따른다. 검색 결과에 없는 정보는 빈 문자열 또는 '확인불가'로 둔다.

{segment_rules}
{results}"""


def verify_company(name: str) -> dict:
    results = search_many([f"{name} 투자 유치 시리즈", f"{name} 상장 IPO"], max_results=5)
    results = [r for r in results if name in r["title"] + r["content"]]  # 해당 기업이 나온 기사만
    if not results:
        return {"company": {"name": name}, "is_eligible": False,
                "eligibility_reason": "상세 확인용 기사를 찾지 못함", "sources": []}

    # 상세 확인은 항목이 많아 nano가 빈칸을 남기므로 mini를 쓴다 (발굴 단계는 nano)
    llm = ChatOpenAI(model=MODEL_ANALYZE, temperature=0).with_structured_output(CompanyProfile)
    p = llm.invoke(VERIFY_PROMPT.format(name=name, results=format_results(results), segment_rules=SEGMENT_RULES))
    ok, reason = is_eligible(p.listed, p.latest_round, p.exited)
    if ok and not p.is_domestic:
        ok, reason = False, "해외 기업"
    return {
        "company": {"name": name, **p.model_dump(), "round": normalize_round(p.latest_round) or p.latest_round},
        "is_eligible": ok,
        "eligibility_reason": reason,
        "sources": [to_source(r, name, "explorer") for r in results],
    }


def explorer_node(state: dict) -> dict:
    candidates = state.get("candidates") or discover_candidates()
    idx = state.get("current_index", 0)
    if idx >= len(candidates):  # 발굴 결과가 비었을 때
        return {**RESET_ON_NEXT, "candidates": candidates, "current_index": idx,
                "is_eligible": False, "eligibility_reason": "조건을 충족하는 후보를 찾지 못함"}
    name = candidates[idx]
    result = verify_company(name)
    out = {**RESET_ON_NEXT, "candidates": candidates, "current_index": idx + 1, **result}
    if not result["is_eligible"]:
        out["rejected"] = [{"company": name, "reason": result["eligibility_reason"], "total": None}]
    return out
