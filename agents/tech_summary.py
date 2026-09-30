"""기술 요약 Self-RAG 노드 (담당: 장정훈).

공통 Chroma tech 문서와 공개 웹 자료를 검색한다. 기업별 주장은 기업명이 명시된
근거에서만 추출하며, 관련성·근거 일치 여부를 평가한 뒤 State에 넣는다.
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


def _format(items: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({item['kind']}) {item['title']}\n{item['text'][:1200]}"
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


def _grounded(name: str, items: list[dict]) -> list[Claim]:
    if not items:
        return []
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
        for i, c in enumerate(candidates)
    )
    grade = _llm(Grounding).invoke(GROUNDING_PROMPT.format(blocks=blocks))
    accepted = {v.claim_id for v in grade.verdicts if v.supported}
    return [c for i, c in enumerate(candidates) if i in accepted]


def _source(item: dict, name: str) -> dict:
    if item["kind"] == "report":
        return report_to_source(item["source"], name, "tech_summary")
    source = web_to_source(item["source"], name, "tech_summary")
    source["source_id"] = f"web:{source['url']}"
    if item["source"].get("source_path"):
        source["source_path"] = item["source"]["source_path"]
    return source


def tech_summary_node(state: dict) -> dict:
    company = state["company"]
    name = company["name"]
    segment = company.get("segment") or "AI 반도체"
    query = f"{name} {segment} AI 반도체 칩 공정 테이프아웃 양산 성능 매출 계약"
    items = _merge(_collect(query, reports=True))
    selected, rewrite = _relevant(name, items)
    company_items = [item for item in selected if _has_company(item, name)]
    if not company_items:  # 관련 근거가 없을 때 검색어를 바꿔 한 번 더 검색
        items = _merge(items + _collect(rewrite or f"{name} AI 칩 제품 계약 매출", reports=False))
        selected, _ = _relevant(name, items)
        company_items = [item for item in selected if _has_company(item, name)]
    evidence = company_items[:18]
    claims = _grounded(name, evidence)
    used_ids = sorted({c.evidence_id for c in claims})
    source_by_id = {i: _source(evidence[i], name) for i in used_ids}
    summary = {field: [c.text for c in claims if c.field == field] for field in FIELDS}
    claim_evidence = []
    for claim in claims:
        source = source_by_id[claim.evidence_id]
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
        claim_evidence.append(trace)
    summary.update({
        "company": name,
        "evidence": claim_evidence,
        "info_insufficient": any(not summary[field] for field in FIELDS),
        "missing_fields": [field for field in FIELDS if not summary[field]],
    })
    sources = [source_by_id[i] for i in used_ids]
    return {"tech_summary": summary, "sources": sources}
