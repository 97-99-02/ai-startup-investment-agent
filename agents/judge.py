"""투자 판단 에이전트 (담당: 김다은)

입력: company, tech_summary, market_analysis, competitor_analysis, team_analysis, sources(현재 기업 것만)
LLM: config.MODEL_JUDGE - 항목별 점수(1~5)와 근거·출처, 법률 리스크 여부, 리스크 목록만 판정 (설계 3.3)
코드: 근거 규칙 적용(출처 없음·정보 부족 → 최대 2점), 환산 점수 계산, config.INVEST_THRESHOLD 비교,
      보류 조건 적용 (설계 3.4, 3.5). 같은 점수에는 항상 같은 결정이 나온다
출력: {"scores": {...}, "rejected": [...](보류일 때)}
  - scores: items({영문 키: {score, evidence, sources, insufficient}}), legal_risk, risks,
    total(환산 점수), decision("투자" | "보류"), hold_reasons, cautions(핵심 항목 2점 이하 주의 코멘트)
    graph.route_after_judge 가 scores["decision"] 으로 분기한다
  - rejected 항목: {"company": 기업명, "reason": 보류 사유, "total": 환산 점수}
"""
import hashlib
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import (CORE_CAUTION_SCORE, CORE_ITEMS, CORE_MIN_SCORE, INVEST_THRESHOLD, LLM_ATTEMPTS, LLM_MAX_TOKENS,
                    LLM_SEED, MODEL_JUDGE, WEIGHTS)

JUDGE_LOG_PATH = Path("outputs/judge_log.jsonl")   # 후보별 채점 로그 (outputs/*는 gitignore)
logger = logging.getLogger(__name__)

# 같은 기업을 다시 평가했을 때 결과가 크게 다르면 재채점한다. 같은 기업 점수가 몇 분 사이에 66~74점을 오가며
# 투자/보류가 뒤집힌 적이 있다. 채점 로그를 평가 기록으로 쓰고, 같은 평가 버전의 기록끼리만 비교한다
# (로직을 바꾸기 전 기록과 비교하면 정상적인 변화도 오차로 잡힌다)
CONSISTENCY_GAP = 5      # 과거 중앙값과 환산 점수가 이만큼 이상 차이 나면 불일치 (결정이 달라도 불일치)
RESCORE_SAMPLES = 2      # 불일치일 때 seed를 바꿔 더 채점하는 횟수 (처음 채점과 합쳐 항목별 중앙값)
_ROOT = Path(__file__).resolve().parent.parent
# 점수에 영향을 주는 파일. 내용이 바뀌면 평가 버전이 바뀐다 (보고서·사실 검증은 점수와 무관해 뺀다)
EVAL_FILES = ("config.py", "agents/explorer.py", "agents/web.py", "agents/tech_summary.py", "agents/market.py",
              "agents/competitor.py", "agents/team.py", "agents/judge.py",
              "rag/ingest.py", "rag/retriever.py", "rag/embeddings.py")

INSUFFICIENT_MAX_SCORE = 2   # 설계 3.5: 공개 정보로 확인할 수 없는 항목은 최대 2점
ITEM_LABELS = {
    "tech": "기술·제품 성숙도",
    "market": "시장성",
    "team": "창업자·팀",
    "traction": "실적·고객 검증",
    "competition": "경쟁 우위",
    "deal": "투자조건",
}

# 설계 3.3 항목별 채점 기준
RUBRIC = """| 항목 | 1점 | 2점 | 3점 | 4점 | 5점 |
|---|---|---|---|---|---|
| tech (기술·제품 성숙도) | 설계·아키텍처 단계 | FPGA 시제품 | 테이프아웃 완료 또는 엔지니어링 샘플 | 양산 시작 | 양산 제품을 복수 고객에 공급 |
| market (시장성) | 세부 분야 시장 수치 없음 | 시장 규모 확인, 연평균 성장률 10% 미만 | 연평균 성장률 10~20% | 연평균 성장률 20% 이상 | 수십억 달러 이상 규모, 연평균 성장률 20% 이상 |
| team (창업자·팀) | 창업자 정보 비공개 | 반도체 외 분야 경력 | 반도체 관련 경력(대기업 반도체 부문, 관련 분야 박사) | 반도체 분야 교수, 대기업 임원급 경력 | 4점 조건에 더해 창업·매각 경험이 있거나 핵심 인력까지 공개됨 |
| traction (실적·고객 검증) | 매출, 고객 없음 | MOU 또는 무상 PoC | 초기 매출(30억 원 미만) 또는 유상 PoC | 매출 30억 원 이상 또는 공급 계약 공개 | 매출 100억 원 이상 또는 대형 공급 계약 |
| competition (경쟁 우위) | 차별점 불분명 | 차별점을 주장하나 수치 근거 없음 | 성능 또는 전력 효율의 수치 우위 제시 | 3점 조건에 더해 특허 또는 자체 소프트웨어 개발 환경 보유 | 4점 조건에 더해 학회 발표 등 제3자 검증 |
| deal (투자조건) | 투자 정보 없음 | 라운드만 공개, 금액 비공개 | 라운드 금액과 투자사 공개 | 기업가치 공개 | 기업가치 공개, 전략적 투자자(대기업 등) 참여 |"""


class ItemScore(BaseModel):
    score: int = Field(description="채점 기준표에 따른 1~5점")
    evidence: str = Field(description="점수의 근거 한두 문장. 분석 자료에 있는 사실만 쓴다")
    source_ids: list[int] = Field(description="근거로 쓴 출처 번호 [n]. 없으면 빈 리스트")
    insufficient: bool = Field(description="공개 정보로 이 항목을 확인할 수 없으면 true")


class LegalRisk(BaseModel):
    unresolved: bool = Field(description="핵심 기술 관련 소송 등 사업 지속에 영향을 주는 법률 리스크가 해소되지 않았으면 true")
    evidence: str = Field(description="판단 근거. 관련 자료가 없으면 '확인된 법률 리스크 없음'")
    source_ids: list[int] = Field(description="근거로 쓴 출처 번호 [n]")


class Risk(BaseModel):
    category: Literal["시장", "기술", "규제", "경쟁", "운영"]
    description: str = Field(description="리스크 한 문장 (분석 자료에 근거)")


class Scorecard(BaseModel):
    tech: ItemScore
    market: ItemScore
    team: ItemScore
    traction: ItemScore
    competition: ItemScore
    deal: ItemScore
    legal_risk: LegalRisk
    risks: list[Risk] = Field(description="시장·기술·규제·경쟁·운영 리스크 3~6개")


PROMPT = """너는 AI 반도체 VC의 투자 심사역이다. '{name}'을 아래 채점 기준표로 항목별 1~5점 채점하라.

[채점 기준표]
{rubric}

[채점 규칙]
1. 분석 자료와 출처에 있는 사실만 근거로 쓴다. 자료에 없는 사실을 추측해 점수를 올리지 마라.
2. 기준표의 한 칸 조건을 모두 충족해야 그 점수를 준다. 두 칸 사이면 낮은 점수를 준다.
3. 근거로 쓴 출처 번호를 source_ids에 넣는다. 출처 목록에 없는 번호는 쓰지 마라.
4. 공개 정보로 항목을 확인할 수 없으면 insufficient=true로 두고, 근거에 '정보 부족'이라고 쓴다.
5. market은 대상 세부 분야 수치(segment_specific=true)만 3~5점 근거로 인정한다.
6. traction은 기술 요약(tech_summary)의 공개 매출·계약만 근거로 쓴다. 투자 라운드·투자사 정보는 traction 근거로 쓰지 않는다.
7. 법률 리스크(legal_risk): 핵심 기술 관련 소송·분쟁이 자료에 있으면 사업 지속에 영향을 주는지 자료로 판단하고, 판단 근거를 evidence에 쓴다.
   소송이 있어도 사업 영향이 없다거나 이미 해소됐다는 출처가 있으면 그 근거를 쓰고 unresolved=false다.
   그런 출처가 없으면 unresolved=true다. 소송·분쟁이 자료에 없으면 unresolved=false다. 판단에 쓴 출처 번호는 source_ids에 넣는다.
8. risks에는 투자 시 고려할 시장·기술·규제·경쟁·운영 리스크를 분석 자료에서 골라 쓴다.
9. 계획·예정·목표·추진 중인 일은 달성한 것으로 인정하지 않는다. 테이프아웃 완료 예정은 테이프아웃 완료가 아니고,
   양산 예정·목표는 양산 시작이 아니며, 계약 논의·협의 중은 계약 체결이 아니다. 완료가 확인된 사실만 해당 점수 조건으로 본다.
10. tech_summary의 기존 7개 항목과 evidence는 기업 웹 사실이고, industry_context는 보고서 기반 업계 해석이다.
    업계 해석이나 보고서의 수치를 해당 기업의 칩·양산·매출·계약 실적으로 인정하지 않는다.
    tech는 기업의 development_stage와 연결된 웹 근거로 판단한다. info_insufficient는 전체 요약의
    누락 여부이므로 공정·약점·계약 등 다른 항목의 누락만으로 tech를 정보 부족 처리하지 않는다.
    industry_context가 없거나 비교 조건이 부족해도 웹으로 확인된 개발 단계의 근거는 유지한다.
    경쟁 우위의 수치 비교는 동일 조건이 확인된 경우만 사용한다.

[분석 자료]
{materials}

[출처 목록]
{sources}"""


def company_sources(state: dict) -> list[dict]:
    """현재 기업 분석에 쓰인 출처만 (이전 후보 출처가 섞이지 않게)."""
    name = state["company"]["name"]
    return [s for s in state.get("sources", []) if s.get("company") == name]


def format_sources(sources: list[dict], limit: int = 300) -> str:
    return "\n".join(
        f"[{i}] ({s.get('node', '')}) {s.get('title', '')} - {s.get('publisher', '')} {s.get('date', '')}"
        f"\n    {s.get('snippet', '')[:limit]}"
        for i, s in enumerate(sources)
    ) or "(출처 없음)"


def cite(ids: list[int], sources: list[dict]) -> list[dict]:
    """출처 번호를 {title, url}로 바꾼다. 목록에 없는 번호는 버린다."""
    return [{"title": sources[i].get("title", ""), "url": sources[i].get("url", "")}
            for i in dict.fromkeys(ids) if 0 <= i < len(sources)]


def is_placeholder(analysis: dict | None) -> bool:
    """분석 결과가 비었거나 아직 구현되지 않은 임시 값({"todo": True})인가."""
    return not analysis or bool(analysis.get("todo"))


def apply_evidence_rules(card: Scorecard, state: dict, sources: list[dict]) -> dict:
    """설계 3.5 근거 규칙을 코드로 적용한다.
    정보 부족이거나 유효한 출처가 없는 항목은 최대 2점으로 제한한다 (근거 없이 높은 점수 방지).
    """
    # 근거를 제공하는 분석 결과가 비어 있으면 LLM이 다른 출처를 인용해도 정보 부족으로 본다
    no_basis = {
        "market": (state.get("market_analysis") or {}).get("info_insufficient", False)
                  or is_placeholder(state.get("market_analysis")),
        "tech": is_placeholder(state.get("tech_summary")),
        "traction": is_placeholder(state.get("tech_summary")),   # 실적 근거는 기술 요약의 공개 매출·계약
        "team": is_placeholder(state.get("team_analysis")),
        "competition": is_placeholder(state.get("competitor_analysis")),
    }
    items = {}
    for key in WEIGHTS:
        s: ItemScore = getattr(card, key)
        cited = cite(s.source_ids, sources)
        insufficient = s.insufficient or not cited or no_basis.get(key, False)
        score = min(max(s.score, 1), 5)   # 스키마 제약 대신 코드로 1~5 보정
        if insufficient:
            score = min(score, INSUFFICIENT_MAX_SCORE)
        evidence = s.evidence
        if insufficient and "정보 부족" not in evidence:
            evidence = f"정보 부족: {evidence}"
        items[key] = {"score": score, "evidence": evidence, "sources": cited, "insufficient": insufficient}
    return items


def total_score(items: dict) -> float:
    """환산 점수 = Σ(항목 점수 ÷ 5 × 항목 비중) (설계 3.1)"""
    return round(sum(items[k]["score"] / 5 * w for k, w in WEIGHTS.items()), 1)


def decide(items: dict, total: float, legal_unresolved: bool) -> tuple[str, list[str]]:
    """설계 3.4 투자 결정 규칙. 보류 사유를 모두 모아 돌려준다."""
    reasons = []
    if total < INVEST_THRESHOLD:
        reasons.append(f"환산 점수 {total:g}점 < 기준 {INVEST_THRESHOLD}점")
    for key in CORE_ITEMS:
        if items[key]["score"] < CORE_MIN_SCORE:
            reasons.append(f"핵심 역량 미달: {ITEM_LABELS[key]} {items[key]['score']}점")
    if legal_unresolved:
        reasons.append("해소되지 않은 법률 리스크")
    return ("보류" if reasons else "투자"), reasons


def build_cautions(items: dict) -> list[dict]:
    """핵심 항목(팀·기술)이 CORE_CAUTION_SCORE 이하이면 주의 코멘트를 만든다. 보류 여부와 별개다.
    정보 부족으로 제한된 점수와 실제로 낮게 평가된 점수를 구분해 적는다. 보고서가 이 목록을 그대로 출력한다.
    """
    cautions = []
    for key in CORE_ITEMS:
        info = items[key]
        if info["score"] > CORE_CAUTION_SCORE:
            continue
        reason = "정보 부족으로 최대 2점 제한" if info["insufficient"] else "낮은 평가"
        # 근거는 스키마상 한두 문장이라 자르지 않는다 (글자 수로 자르면 '…단계로 볼'처럼 문장 중간에서 끊겼다)
        basis = info["evidence"].removeprefix("정보 부족:").strip()
        cautions.append({
            "item": key,
            "label": ITEM_LABELS[key],
            "score": info["score"],
            "insufficient": info["insufficient"],
            "comment": f"{ITEM_LABELS[key]} {info['score']}점 ({reason}): {basis}",
        })
    return cautions


def eval_version() -> str:
    """점수에 영향을 주는 파일 내용의 해시. 커밋하지 않은 수정도 구분된다."""
    digest = hashlib.sha256()
    for rel in EVAL_FILES:
        path = _ROOT / rel
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def past_records(name: str, version: str) -> list[dict]:
    """채점 로그에서 같은 기업·같은 평가 버전의 기록. 로그가 없거나 읽을 수 없으면 빈 목록."""
    try:
        lines = JUDGE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("company") == name and r.get("eval_version") == version:
            records.append(r)
    return records


def is_inconsistent(total: float, decision: str, history: list[dict]) -> bool:
    """과거 기록의 다수 결정과 다르거나, 과거 환산 점수 중앙값과 CONSISTENCY_GAP 이상 차이 나면 True."""
    if not history:
        return False
    decisions = [r["decision"] for r in history]
    # 과반인 결정이 있을 때만 결정을 비교한다 (투자 1회·보류 1회처럼 동률이면 점수 차이만 본다)
    majority = next((d for d in set(decisions) if decisions.count(d) * 2 > len(decisions)), None)
    changed = majority is not None and decision != majority
    return changed or abs(total - statistics.median(r["total"] for r in history)) >= CONSISTENCY_GAP


def median_card(cards: list[Scorecard]) -> Scorecard:
    """항목마다 점수의 중앙값을 고르고, 그 점수를 준 채점의 근거·출처를 함께 쓴다.
    법률 리스크는 다수 판정, 리스크 목록은 처음 채점 것을 쓴다."""
    middle = len(cards) // 2
    picked = {key: sorted((getattr(c, key) for c in cards), key=lambda s: s.score)[middle] for key in WEIGHTS}
    picked["legal_risk"] = sorted((c.legal_risk for c in cards), key=lambda x: x.unresolved)[middle]
    return cards[0].model_copy(update=picked)


def write_log(name: str, card: Scorecard, scores: dict, version: str = "", consistency: dict | None = None) -> None:
    """후보 1개당 JSON 한 줄을 append한다. State에는 넣지 않는다. 실패해도 그래프는 멈추지 않는다."""
    try:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "company": name,
            "items": {
                k: {
                    "raw_score": getattr(card, k).score,   # LLM이 준 원래 점수
                    "score": v["score"],                    # 근거 규칙 적용 뒤 최종 점수
                    "insufficient": v["insufficient"],
                    "evidence": v["evidence"][:80],
                }
                for k, v in scores["items"].items()
            },
            "legal_risk": {
                "raw_unresolved": card.legal_risk.unresolved,          # LLM이 준 원래 판정
                "unresolved": scores["legal_risk"]["unresolved"],      # 출처 규칙 적용 뒤 (보류 사유가 되는 값)
                "evidence": scores["legal_risk"]["evidence"][:200],
                "sources": [x["title"] for x in scores["legal_risk"]["sources"]],
            },
            "total": scores["total"],
            "decision": scores["decision"],
            "hold_reasons": scores["hold_reasons"],
            "cautions": [c["comment"] for c in scores["cautions"]],
            "eval_version": version,
            "consistency": consistency or {},
        }
        JUDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JUDGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 - 로그 실패로 평가 흐름을 끊지 않는다
        logger.warning("채점 로그를 쓰지 못했습니다: %s", e)


def _judge_llm(seed: int):
    return ChatOpenAI(model=MODEL_JUDGE, temperature=0, seed=seed, max_tokens=LLM_MAX_TOKENS).with_structured_output(Scorecard).with_retry(stop_after_attempt=LLM_ATTEMPTS)


def score_card(card: Scorecard, state: dict, sources: list[dict]) -> dict:
    """채점 결과에 근거 규칙·환산 점수·결정 규칙을 적용해 scores를 만든다."""
    items = apply_evidence_rules(card, state, sources)
    total = total_score(items)
    legal_sources = cite(card.legal_risk.source_ids, sources)
    legal_unresolved = card.legal_risk.unresolved and bool(legal_sources)  # 출처 없는 법률 리스크로는 보류하지 않음
    decision, hold_reasons = decide(items, total, legal_unresolved)
    return {
        "items": items,
        "legal_risk": {
            "unresolved": legal_unresolved,
            "evidence": card.legal_risk.evidence,
            "sources": legal_sources,
        },
        "risks": [r.model_dump() for r in card.risks],
        "total": total,
        "decision": decision,
        "hold_reasons": hold_reasons,
        "cautions": build_cautions(items),
    }


def judge_node(state: dict) -> dict:
    name = state["company"]["name"]
    sources = company_sources(state)
    materials = {k: state.get(k) for k in
                 ("company", "tech_summary", "market_analysis", "competitor_analysis", "team_analysis")}
    if isinstance(materials.get("tech_summary"), dict):
        materials["tech_summary"] = {k: v for k, v in materials["tech_summary"].items()
                                    if k != "diagnostics"}
    prompt = PROMPT.format(
        name=name, rubric=RUBRIC,
        materials=json.dumps(materials, ensure_ascii=False, indent=1),
        sources=format_sources(sources),
    )
    card = _judge_llm(LLM_SEED).invoke(prompt)
    scores = score_card(card, state, sources)

    version = eval_version()
    history = past_records(name, version)
    consistency = {"history": len(history), "rescored": False}
    if history:
        consistency["past_median_total"] = statistics.median(r["total"] for r in history)
    if is_inconsistent(scores["total"], scores["decision"], history):
        cards = [card] + [_judge_llm(LLM_SEED + i).invoke(prompt) for i in range(1, RESCORE_SAMPLES + 1)]
        card = median_card(cards)
        consistency.update(rescored=True, first_total=scores["total"],
                           sample_totals=[score_card(c, state, sources)["total"] for c in cards])
        scores = score_card(card, state, sources)
        # 재채점 뒤에도 다르면 분석 입력(검색 결과 등)이 바뀐 경우라 기록으로 남긴다
        consistency["still_inconsistent"] = is_inconsistent(scores["total"], scores["decision"], history)
    write_log(name, card, scores, version, consistency)
    out = {"scores": scores}
    if scores["decision"] == "보류":
        out["rejected"] = [{"company": name, "reason": "; ".join(scores["hold_reasons"]), "total": scores["total"]}]
    return out
