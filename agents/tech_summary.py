"""기술 요약 Self-RAG 노드 (담당: 장정훈).

기업별 칩·공정·개발 단계·성능·매출·계약은 공개 웹 자료에서 수집한다.
Chroma tech 보고서는 업계 기준과 기술적 의미를 해석하는 데 사용한다.
각 기업 사실과 업계 해석의 근거를 따로 검증한 뒤 State에 넣는다.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.web import search_many, to_source as web_to_source
from config import MODEL_ANALYZE
from rag.retriever import get_retriever, to_source as report_to_source

logger = logging.getLogger(__name__)

FIELDS = ("core_chips", "process", "development_stage", "performance_metrics",
          "strengths", "weaknesses", "public_revenue_contracts")
FieldName = Literal["core_chips", "process", "development_stage", "performance_metrics",
                    "strengths", "weaknesses", "public_revenue_contracts"]


class Relevance(BaseModel):
    relevant_ids: list[int] = Field(description="기술·제품·실적을 확인하는 데 쓸 수 있는 근거 번호")
    rewrite_query: str = Field(description="관련 근거가 부족할 때 재검색할 짧은 검색어")


class Claim(BaseModel):
    field: FieldName
    text: str = Field(description="근거가 직접 뒷받침하는 한 가지 사실")
    evidence_id: int
    quote: str = Field(description="근거에서 글자 그대로 옮긴 짧은 부분")


class Draft(BaseModel):
    claims: list[Claim]


class Verdict(BaseModel):
    claim_id: int
    supported: bool


class Grounding(BaseModel):
    verdicts: list[Verdict]


class IndustryInsight(BaseModel):
    topic: Literal["performance", "process", "maturity", "tradeoffs"]
    text: str = Field(description="기업 사실을 보고서의 업계 기준으로 해석한 문장")
    company_fact_ids: list[int] = Field(description="해석 대상 기업 사실 번호")
    report_id: int = Field(description="업계 기준의 근거 보고서 번호")
    quote: str = Field(description="보고서에서 그대로 복사한 근거")
    comparison: Literal["context_only", "comparable", "conditions_missing"] = Field(
        description="일반적 해석 / 동일 조건 수치 비교 / 비교 조건 부족")


class IndustryDraft(BaseModel):
    insights: list[IndustryInsight]


def _llm(schema):
    return ChatOpenAI(model=MODEL_ANALYZE, temperature=0).with_structured_output(schema)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _has_company(item: dict, name: str) -> bool:
    return _compact(name) in _compact(item["title"] + " " + item["text"])


def _key(item: dict) -> tuple:
    if item["kind"] == "web":
        return ("web", item["source"]["url"])
    doc = item["source"]
    return ("report", doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


def _merge(items: list[dict]) -> list[dict]:
    merged, seen = [], set()
    for item in items:
        if _key(item) not in seen:
            merged.append(item)
            seen.add(_key(item))
    return merged


<<<<<<< Updated upstream
def _collect(query: str, *, reports: bool) -> list[dict]:
    items = []
    if reports:
        for doc in get_retriever("tech", k=5).invoke(query):
            items.append({"kind": "report", "source": doc, "text": doc.page_content[:800],
                          "title": doc.metadata.get("title", "")})
    try:
        results = search_many([query], max_results=8, topic="general")
    except Exception as exc:
        logger.warning("기술 요약 웹 검색 실패: %s", exc)
        results = []
    for result in results:
        items.append({"kind": "web", "source": result, "text": result.get("content", "")[:800],
                      "title": result.get("title", "")})
    return items
=======
WEB_TEXT_LIMIT = 6000
MAX_COMPANY_EVIDENCE = 24
SEARCH_GROUPS = (
    (("core_chips", "process", "development_stage"), "칩 제품 공정 테이프아웃 샘플 양산 현황"),
    (("performance_metrics", "strengths", "weaknesses"), "성능 TOPS 전력 효율 벤치마크 소프트웨어 한계"),
    (("public_revenue_contracts",), "매출 실적 유상 공급 계약 고객"),
)


def _collect_web(queries: list[str]) -> list[dict]:
    # 검색 오류를 '공개 정보 없음'으로 바꾸지 않는다. 호출자가 실패 원인을 확인할 수 있다.
    results = search_many(queries, max_results=4, topic="general")
    return [{"kind": "web", "source": r,
             "text": (r.get("raw_content") or r.get("content", ""))[:WEB_TEXT_LIMIT],
             "title": r.get("title", ""), "date": source_date(r)} for r in results]


RETRY_TERMS = {
    "core_chips": "칩 제품명 데이터시트",
    "process": "제조 파운드리 공정 nm",
    "development_stage": "테이프아웃 완료 엔지니어링 샘플 양산 시작",
    "performance_metrics": "TOPS/W 지연시간 정밀도 성능 측정",
    "strengths": "기술 장점 SDK 소프트웨어",
    "weaknesses": "기술 한계 지원 미지원 독립 검증",
    "public_revenue_contracts": "연도별 매출 실적 유상 고객 공급계약 체결",
}


def _queries(name: str, segment: str, missing: list[str], *, retry: bool = False) -> list[str]:
    if retry:
        return [f"{name} {segment} " + " ".join(RETRY_TERMS[f] for f in fields if f in missing)
                for fields, _ in SEARCH_GROUPS if set(fields).intersection(missing)]
    return [f"{name} {segment} {terms}" for fields, terms in SEARCH_GROUPS
            if set(fields).intersection(missing)]
>>>>>>> Stashed changes


def _format(items: list[dict]) -> str:
    return "\n\n".join(
<<<<<<< Updated upstream
        f"[{i}] ({item['kind']}) {item['title']}\n{item['text'][:1200]}"
=======
        f"[{i}] ({item['kind']}, {_date_label(item)}) {item['title']}\n{item['text']}"
>>>>>>> Stashed changes
        for i, item in enumerate(items)
    )


RELEVANCE_PROMPT = """'{name}'의 핵심 칩, 공정, 개발·양산 단계, 성능 지표, 강점·약점,
공개 매출·계약을 확인하는 데 유용한 근거 번호를 고르세요. 무관한 조각은 제외하세요.
관련 근거가 부족하면 재검색어를 작성하세요.

{evidence}"""

DRAFT_PROMPT = """'{name}'에 대해 아래 근거에서 확인되는 사실만 claims로 추출하세요.
각 claim에 field, text, evidence_id, quote를 넣으세요. quote는 원문을 그대로 복사하세요.
숫자와 단위를 바꾸거나 추정하지 마세요. 기업명이 조각에 명시된 경우에만 기업의
칩·공정·단계·성능·강점·약점·매출·계약을 주장하세요. 산업 일반 보고서를 기업 실적으로
연결하지 마세요. 계획과 양산 완료, 투자 유치와 매출, 협력 논의와 체결 계약을 구분하세요.
확인되지 않은 항목은 생략하세요.

{evidence}"""

GROUNDING_PROMPT = """각 주장을 지정된 원문과 대조하세요. 원문이 주체·상태·수치·시점을
직접 뒷받침할 때만 supported=true입니다. 과장하거나 계획을 실적으로 바꾸면 false입니다.
모든 claim_id에 답하세요.

{blocks}"""


def _relevant(name: str, items: list[dict]) -> tuple[list[dict], str]:
    if not items:
        return [], ""
    result = _llm(Relevance).invoke(RELEVANCE_PROMPT.format(name=name, evidence=_format(items)))
    selected = [items[i] for i in dict.fromkeys(result.relevant_ids) if 0 <= i < len(items)]
    return selected, result.rewrite_query.strip()


def _grounded(name: str, items: list[dict], audit: list[dict] | None = None) -> list[Claim]:
    if not items:
        return []
<<<<<<< Updated upstream
    draft = _llm(Draft).invoke(DRAFT_PROMPT.format(name=name, evidence=_format(items)))
    candidates = [
        c for c in draft.claims
        if 0 <= c.evidence_id < len(items)
        and c.quote.strip()
        and _compact(c.quote) in _compact(items[c.evidence_id]["text"])
        and _has_company(items[c.evidence_id], name)
    ]
    if not candidates:
        return []
    blocks = "\n\n".join(
        f"[{i}] 주장: {c.text}\n원문: {items[c.evidence_id]['text'][:1200]}"
=======
    draft = _llm(Draft).invoke(DRAFT_PROMPT.format(name=name, evidence=_format(items),
                                                today=date.today().isoformat()))
    candidates = []
    for c in draft.claims:
        reason = ""
        if not 0 <= c.evidence_id < len(items):
            reason = "invalid_evidence_id"
        else:
            item = items[c.evidence_id]
            if item["kind"] != "web":
                reason = "company_fact_requires_web"
            elif not _has_company(item, name):
                reason = "company_not_mentioned"
            elif not c.quote.strip() or _compact(c.quote) not in _compact(item["text"]):
                reason = "quote_not_in_source"
            elif not _time_ok(c, item):
                reason = "unverified_date"
        if reason:
            if audit is not None:
                audit.append({"field": c.field, "text": c.text, "reason": reason})
        else:
            candidates.append(_with_time(c, items[c.evidence_id]))
    if not candidates:
        return []
    blocks = "\n\n".join(
        f"[{i}] 주장: {c.text}\n원문({_date_label(items[c.evidence_id])}): {items[c.evidence_id]['text']}"
>>>>>>> Stashed changes
        for i, c in enumerate(candidates)
    )
    grade = _llm(Grounding).invoke(GROUNDING_PROMPT.format(blocks=blocks))
    accepted = {v.claim_id for v in grade.verdicts if v.supported}
    if audit is not None:
        audit.extend({"field": c.field, "text": c.text, "reason": "grounding_not_supported"}
                     for i, c in enumerate(candidates) if i not in accepted)
    return [c for i, c in enumerate(candidates) if i in accepted]


def _source(item: dict, name: str) -> dict:
    if item["kind"] == "report":
        source = report_to_source(item["source"], name, "tech_summary")
        source["snippet"] = item["text"]
        return source
    source = web_to_source(item["source"], name, "tech_summary")
    source["source_id"] = f"web:{source['url']}"
    source["snippet"] = item["text"]
    if item["source"].get("source_path"):
        source["source_path"] = item["source"]["source_path"]
    return source


INDUSTRY_PROMPT = """기업 '{name}'의 검증된 웹 사실을 아래 기술 보고서의 업계 기준으로 해석하세요.
기업 사실은 웹 사실 목록에서만 가져오고, 보고서는 업계 맥락에만 사용하세요.
기업명이 없는 일반 보고서도 업계 기준 근거로 사용할 수 있습니다.
각 해석에 company_fact_ids, report_id, 보고서 원문 quote를 붙이세요.
- 공정 세대의 의미, 성능 평가에 필요한 조건, 개발 단계의 의미, 기술적 강점·제약을 설명하세요.
- TOPS/W 등 수치 우위는 지표·단위·정밀도·워크로드·전력 측정 범위가 양쪽에서 확인되고
  동일한 경우에만 comparison=comparable입니다. 조건이 빠지면 conditions_missing으로 두고
  조건 부족을 명시하세요. 일반적 의미만 설명하면 context_only입니다.
- 보고서의 수치를 기업 실적으로 바꾸거나, 서로 다른 칩/제품의 수치를 결합하지 마세요.
- 기업이나 보고서 근거에 없는 칩·공정·양산·매출·계약을 추가하지 마세요.
- 근거 없는 업계 순위·기술 우위나 날짜를 추정하지 마세요. 유용한 해석이 없으면 빈 목록입니다.

[기업 웹 사실]
{facts}

[업계 보고서]
{reports}"""

INDUSTRY_GROUNDING_PROMPT = """아래 해석을 기업 웹 사실과 보고서 원문에 대조하세요.
두 근거가 함께 뒷받침할 때만 supported=true입니다. 보고서 내용을 기업의 실적으로
바꾸거나 수치 우위를 추정하면 false입니다. comparable은 지표·단위·정밀도·워크로드·
전력 측정 범위가 양쪽에서 동일하다고 확인되는 경우만 허용합니다.
조건이 부족하다는 해석은 그 부족함을 명시하면 허용합니다. 모든 claim_id에 답하세요.

{blocks}"""


def _industry_context(name: str, segment: str, claims: list[Claim],
                      evidence: list[dict]) -> tuple[list[dict], list[dict], str, dict]:
    if not claims:
        return [], [], "no_company_facts", {"retrieved_reports": 0}
    # 회사명 대신 세부 분야와 확인된 기술 수치로 일반 기술 보고서를 검색한다.
    technical_facts = " ".join(c.text for c in claims
                               if c.field in ("process", "performance_metrics", "development_stage"))
    technical_facts = re.sub(re.escape(name), "", technical_facts, flags=re.IGNORECASE)
    query = f"{segment} AI 반도체 공정 세대 성능 전력 효율 TOPS/W 개발 단계 {technical_facts}"
    try:
        docs = get_retriever("tech", k=5).invoke(query)
    except FileNotFoundError as exc:
        logger.warning("기술 요약 업계 RAG 사용 불가: %s", exc)
        return [], [], "retrieval_unavailable", {"query": query, "error": str(exc), "retrieved_reports": 0}
    reports = _merge([{"kind": "report", "source": d, "text": d.page_content,
                       "title": d.metadata.get("title", ""), "date": d.metadata.get("year", "")}
                      for d in docs])
    diagnostics = {"query": query, "retrieved_reports": len(reports)}
    if not reports:
        return [], [], "no_reports", diagnostics
    facts = "\n".join(f"[{i}] {c.field}: {c.text} (웹 원문: {c.quote})" for i, c in enumerate(claims))
    draft = _llm(IndustryDraft).invoke(INDUSTRY_PROMPT.format(
        name=name, facts=facts, reports=_format(reports)))
    candidates = [v for v in draft.insights
                  if 0 <= v.report_id < len(reports)
                  and v.company_fact_ids
                  and all(0 <= i < len(claims) for i in v.company_fact_ids)
                  and v.quote.strip()
                  and _compact(v.quote) in _compact(reports[v.report_id]["text"])]
    diagnostics.update({"draft_insights": len(draft.insights),
                        "invalid_insights": len(draft.insights) - len(candidates)})
    if not candidates:
        return [], [], "no_supported_context", diagnostics
    blocks = "\n\n".join(
        f"[{i}] 해석: {v.text}\n비교 상태: {v.comparison}\n기업 사실: "
        + " | ".join(f"{claims[j].text} (원문: {claims[j].quote})" for j in v.company_fact_ids)
        + f"\n보고서({_date_label(reports[v.report_id])}): {reports[v.report_id]['text']}"
        for i, v in enumerate(candidates))
    grade = _llm(Grounding).invoke(INDUSTRY_GROUNDING_PROMPT.format(blocks=blocks))
    supported = {v.claim_id for v in grade.verdicts if v.supported}
    accepted = [v for i, v in enumerate(candidates) if i in supported]
    source_by_id = {i: _source(reports[i], name) for i in sorted({v.report_id for v in accepted})}
    context = []
    for v in accepted:
        source = source_by_id[v.report_id]
        context.append({"topic": v.topic, "text": v.text, "comparison": v.comparison,
                        "company_source_ids": list(dict.fromkeys(
                            _source(evidence[claims[i].evidence_id], name)["source_id"]
                            for i in v.company_fact_ids)),
                        "report_source_id": source["source_id"], "quote": v.quote,
                        "title": source["title"], "date": source.get("date", ""),
                        "source_file": source.get("source_file", ""),
                        "source_path": source.get("source_path", ""), "page": source.get("page")})
    diagnostics["unsupported_insights"] = len(candidates) - len(accepted)
    return context, list(source_by_id.values()), "available" if context else "no_supported_context", diagnostics


def tech_summary_node(state: dict) -> dict:
    company = state["company"]
    name = company["name"]
    segment = company.get("segment") or "AI 반도체"
<<<<<<< Updated upstream
    query = f"{name} {segment} AI 반도체 칩 공정 테이프아웃 양산 성능 매출 계약"
    items = _merge(_collect(query, reports=True))
=======
    queries = _queries(name, segment, list(FIELDS))
    items = _merge(_collect_web(queries))
>>>>>>> Stashed changes
    selected, rewrite = _relevant(name, items)
    evidence = [item for item in selected if _has_company(item, name)][:MAX_COMPANY_EVIDENCE]
    rejected = []
    claims = _grounded(name, evidence, rejected)
    missing = [field for field in FIELDS if not any(c.field == field for c in claims)]
    retry_queries = _queries(name, segment, missing, retry=True) if missing else []
    if not claims and rewrite and rewrite not in queries + retry_queries:
        retry_queries.append(rewrite)
    if retry_queries:
        more = _merge(_collect_web(retry_queries))
        seen = {_key(item) for item in items}
        new_items = [item for item in more if _key(item) not in seen]
        items = _merge(items + more)
        selected, _ = _relevant(name, new_items)
        extra = [item for item in selected if _has_company(item, name)]
        extra = extra[:max(0, MAX_COMPANY_EVIDENCE - len(evidence))]
        new_claims = _grounded(name, extra, rejected)
        offset = len(evidence)
        claims.extend(c.model_copy(update={"evidence_id": c.evidence_id + offset}) for c in new_claims)
        evidence.extend(extra)
    # 초기 검색에서 검증한 사실은 재검색 후에도 보존한다.
    unique = {}
    for c in claims:
        unique.setdefault((c.field, _compact(c.text)), c)
    claims = list(unique.values())
    used_ids = sorted({c.evidence_id for c in claims})
    source_by_id = {i: _source(evidence[i], name) for i in used_ids}
    summary = {field: [c.text for c in claims if c.field == field] for field in FIELDS}
    claim_evidence = []
    for claim in claims:
        source = source_by_id[claim.evidence_id]
<<<<<<< Updated upstream
        trace = {
            "field": claim.field,
            "text": claim.text,
            "quote": claim.quote,
            "source_id": source["source_id"],
            "kind": source["kind"],
            "title": source["title"],
        }
        if source["kind"] == "report":
            trace.update({key: source[key] for key in ("source_file", "source_path", "page")
                          if key in source})
        else:
            trace["url"] = source["url"]
            if source.get("source_path"):
                trace["source_path"] = source["source_path"]
=======
        trace = {"field": claim.field, "text": claim.text, "quote": claim.quote,
                 "source_id": source["source_id"], "kind": "web", "title": source["title"],
                 "date": source.get("date", ""), "url": source["url"]}
        if source.get("source_path"):
            trace["source_path"] = source["source_path"]
>>>>>>> Stashed changes
        claim_evidence.append(trace)
    context, report_sources, context_status, rag_diagnostics = _industry_context(name, segment, claims, evidence)
    summary.update({
        "company": name,
        "evidence": claim_evidence,
        "info_insufficient": any(not summary[field] for field in FIELDS),
        "missing_fields": [field for field in FIELDS if not summary[field]],
        "field_status": {field: "supported" if summary[field] else "not_found" for field in FIELDS},
        "industry_context": context,
        "industry_context_status": context_status,
        "diagnostics": {"web_queries": queries, "retry_queries": retry_queries,
                        "retrieved_web": len(items), "selected_web": len(evidence),
                        "accepted_claims": len(claims), "rejected_claims": rejected,
                        "rag": rag_diagnostics},
    })
    sources = [source_by_id[i] for i in used_ids] + report_sources
    return {"tech_summary": summary, "sources": sources}
