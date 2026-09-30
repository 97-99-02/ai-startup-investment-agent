"""기업 웹 사실 + 업계 RAG 해석 Self-RAG와 KURE-v1 검색 테스트.

기본 테스트: uv run python -m unittest discover -s tests -p 'test_tech_summary.py' -v
실제 임베딩/Chroma 테스트: RUN_KURE_INTEGRATION=1 uv run python -m unittest discover -s tests -p 'test_tech_summary.py' -v
기본 테스트를 실행하면 더미 기술 요약을 터미널에 출력하고 outputs/에도 저장한다.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from agents import tech_summary as tech
from config import EMBEDDING_MODEL
from rag import embeddings

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "tech_summary_dummy.json").read_text(encoding="utf-8")
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fixture_document(entry: dict) -> Document:
    path = PROJECT_ROOT / entry["source_path"]
    return Document(page_content=path.read_text(encoding="utf-8"),
                    metadata={**entry["metadata"], "source_path": entry["source_path"]})


def fixture_web_result(entry: dict) -> dict:
    path = PROJECT_ROOT / entry["source_path"]
    return {**entry, "content": path.read_text(encoding="utf-8")}


class Chain:
    def __init__(self, response):
        self.response = response

    def invoke(self, _prompt):
        return self.response


class TechSummaryTests(unittest.TestCase):
    def setUp(self):
        self.report = Document(
            page_content="AI 반도체는 메모리 대역폭의 영향을 받는다.",
            metadata={"title": "일반 기술 보고서", "source": "general.pdf", "page": 1},
        )
        self.web = {
            "title": "테스트칩 NPU-A 시제품 공개",
            "url": "https://example.org/chip",
            "content": "테스트칩은 NPU-A 시제품을 공개했다.",
        }

    def test_kure_configuration_and_embedding_calls(self):
        self.assertEqual(EMBEDDING_MODEL, "nlpai-lab/KURE-v1")
        model = MagicMock()
        model.prompts = {}
        model.encode.side_effect = [
            MagicMock(tolist=lambda: [[0.1, 0.2]]),
            MagicMock(tolist=lambda: [0.1, 0.2]),
        ]
        with (
            patch.object(embeddings, "SentenceTransformer", return_value=model) as factory,
            patch.object(embeddings, "_device", return_value="cpu"),
        ):
            embedder = embeddings.LocalEmbeddings()
            self.assertEqual(embedder.embed_documents(["기술 문서"]), [[0.1, 0.2]])
            self.assertEqual(embedder.embed_query("NPU 성능"), [0.1, 0.2])
        factory.assert_called_once_with("nlpai-lab/KURE-v1", device="cpu")
        self.assertEqual(model.encode.call_args_list[0].kwargs,
                         {"batch_size": 8, "normalize_embeddings": True})
        self.assertEqual(model.encode.call_args_list[1].kwargs,
                         {"normalize_embeddings": True})

    def test_web_fact_and_company_free_report_have_different_roles(self):
        insight = tech.IndustryInsight(
            topic="tradeoffs", text="시제품의 추론 성능을 평가할 때 메모리 대역폭도 확인해야 한다.",
            company_fact_ids=[0], report_id=0, quote="메모리 대역폭의 영향을 받는다",
            comparison="context_only")
        with (
            patch.object(tech, "get_retriever") as retriever,
            patch.object(tech, "search_many", return_value=[self.web]) as search,
            patch.object(tech, "_llm") as llm,
        ):
            retriever.return_value.invoke.return_value = [self.report]
            llm.side_effect = lambda schema: Chain({
                tech.Relevance: tech.Relevance(relevant_ids=[0], rewrite_query=""),
                tech.Draft: tech.Draft(claims=[
                    tech.Claim(field="core_chips", text="NPU-A 시제품 공개",
                               evidence_id=0, quote="NPU-A 시제품을 공개했다"),
                ]),
                tech.IndustryDraft: tech.IndustryDraft(insights=[insight]),
                tech.Grounding: tech.Grounding(
                    verdicts=[tech.Verdict(claim_id=0, supported=True)]),
            }[schema])
            result = tech.tech_summary_node({"company": {"name": "테스트칩"}})
        summary = result["tech_summary"]
        self.assertEqual(summary["core_chips"], ["NPU-A 시제품 공개"])
        self.assertEqual(summary["industry_context"][0]["text"], insight.text)
        self.assertEqual(summary["industry_context_status"], "available")
        self.assertEqual([s["kind"] for s in result["sources"]], ["web", "report"])
        self.assertEqual(summary["evidence"][0]["kind"], "web")
        self.assertEqual(summary["industry_context"][0]["company_source_ids"],
                         [summary["evidence"][0]["source_id"]])
        self.assertEqual(summary["industry_context"][0]["report_source_id"],
                         result["sources"][1]["source_id"])
        # 첫 검색에서 기업 자료가 있어도 빈 항목에 대해 한 번 더 검색한다.
        self.assertEqual(search.call_count, 2)
        retriever.assert_called_once_with("tech", k=5)
        self.assertNotIn("테스트칩", retriever.return_value.invoke.call_args.args[0])

    def test_dummy_data_covers_web_facts_and_rag_interpretation(self):
        docs = [fixture_document(entry) for entry in FIXTURE["documents"]
                if entry["metadata"]["source"] == "industry_context.md"]
        claims = [
            tech.Claim(field="core_chips", text=FIXTURE["expected"]["core_chips"],
                       evidence_id=0, quote="엣지 추론용 NPU 가온-X1을 개발했다"),
            tech.Claim(field="process", text=FIXTURE["expected"]["process"],
                       evidence_id=0, quote="7nm 공정으로 설계되었고"),
            tech.Claim(field="development_stage", text=FIXTURE["expected"]["development_stage"],
                       evidence_id=0, quote="2026년 3월 테이프아웃을 완료했다"),
            tech.Claim(field="performance_metrics", text=FIXTURE["expected"]["performance_metrics"],
                       evidence_id=1, quote="24 TOPS, 소비 전력을 8 W"),
            tech.Claim(field="strengths", text=FIXTURE["expected"]["strengths"],
                       evidence_id=0, quote="저전력 설계를 강점으로 제시했다"),
            tech.Claim(field="weaknesses", text=FIXTURE["expected"]["weaknesses"],
                       evidence_id=1, quote="독립 검증 결과는 공개되지 않아"),
            tech.Claim(field="public_revenue_contracts",
                       text=FIXTURE["expected"]["public_revenue_contracts"],
                       evidence_id=2, quote="2025년 공개 매출 12억 원을 발표했다. 2026년 4월 시제품 300개 공급계약을 체결했다고 밝혔다"),
        ]
        context = FIXTURE["expected_context"]
        with (
            patch.object(tech, "get_retriever") as retriever,
            patch.object(tech, "search_many",
                         return_value=[fixture_web_result(e) for e in FIXTURE["web_results"]]) as search,
            patch.object(tech, "_llm") as llm,
        ):
            retriever.return_value.invoke.return_value = docs
            llm.side_effect = lambda schema: Chain({
                tech.Relevance: tech.Relevance(relevant_ids=[0, 1, 2], rewrite_query=""),
                tech.Draft: tech.Draft(claims=claims),
                tech.IndustryDraft: tech.IndustryDraft(insights=[tech.IndustryInsight(
                    **context, company_fact_ids=[3], report_id=0), tech.IndustryInsight(
                    **FIXTURE["expected_process_context"], company_fact_ids=[1], report_id=0)]),
                tech.Grounding: tech.Grounding(verdicts=[
                    tech.Verdict(claim_id=i, supported=True) for i in range(len(claims))
                ]),
            }[schema])
            result = tech.tech_summary_node({"company": FIXTURE["company"]})
        summary = result["tech_summary"]
        for field, value in FIXTURE["expected"].items():
            self.assertEqual(summary[field], [value])
        self.assertFalse(summary["info_insufficient"])
        self.assertEqual(summary["missing_fields"], [])
        self.assertEqual(len(result["sources"]), 4)
        self.assertEqual(search.call_count, 1)
        self.assertTrue(any("가온-X1" in q for q in search.call_args.args[0]))
        self.assertEqual(summary["industry_context"][0]["text"], context["text"])
        self.assertEqual(summary["industry_context"][0]["comparison"], "conditions_missing")
        self.assertEqual(summary["industry_context"][0]["report_source_id"], result["sources"][-1]["source_id"])
        self.assertEqual(summary["industry_context"][1]["topic"], "process")
        self.assertEqual(summary["industry_context"][1]["text"], FIXTURE["expected_process_context"]["text"])
        self.assertTrue(all(e["kind"] == "web" for e in summary["evidence"]))
        self.assertNotIn("chunk_id", summary["evidence"][0])
        self.assertTrue(all((PROJECT_ROOT / source["source_path"]).is_file()
                            for source in result["sources"]))
        self.assertEqual(summary["evidence"][3]["url"], "https://example.org/fictional/performance")
        self.write_dummy_result(result)

    def write_dummy_result(self, result):
        output = PROJECT_ROOT / "outputs" / "tech_summary_dummy_result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "notice": "가상 데이터와 모의 LLM 응답으로 만든 테스트 결과입니다. 실제 기업 분석 결과가 아닙니다.",
            **result,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        labels = {
            "core_chips": "핵심 칩", "process": "공정", "development_stage": "개발 단계",
            "performance_metrics": "성능 지표", "strengths": "강점",
            "weaknesses": "약점", "public_revenue_contracts": "공개 매출·계약",
        }
        lines = [f"# {FIXTURE['company']['name']} 기술 요약 (더미 테스트)", "",
                 "> 가상 자료와 모의 LLM 응답으로 만든 테스트 결과입니다. 실제 기업 분석 결과가 아닙니다.",
                 "", "## 기업 정보 — 웹 검색"]
        for field, label in labels.items():
            lines.extend(["", f"### {label}", *[f"- {v}" for v in result["tech_summary"][field]]])
            for trace in result["tech_summary"]["evidence"]:
                if trace["field"] == field:
                    lines.append(f"  - 웹 근거: {trace['title']} · \"{trace['quote']}\"")
        lines.extend(["", "## 업계 기준 해석 — RAG"])
        for insight in result["tech_summary"]["industry_context"]:
            lines.extend([f"- {insight['text']}", f"  - 비교 상태: `{insight['comparison']}`",
                          f"  - 보고서 근거: {insight['title']} p.{insight['page']} · \"{insight['quote']}\""])
        lines.extend(["", "## 사용한 출처"])
        for source in result["sources"]:
            local_file = (PROJECT_ROOT / source["source_path"]).resolve()
            lines.append(f"- [{source['title']}]({local_file}) ({source['kind']}, 원본 파일)")
        markdown = "\n".join(lines) + "\n"
        output.with_suffix(".md").write_text(markdown, encoding="utf-8")
        print(f"\n{markdown}결과 파일: {output.with_suffix('.md')}\n", flush=True)

    def test_missing_development_stage_is_retried_and_initial_fact_is_preserved(self):
        stage = {"title": "테스트칩 테이프아웃 완료", "url": "https://example.org/stage",
                 "content": "테스트칩은 2026년 3월 테이프아웃을 완료했다."}
        drafts = iter([
            tech.Draft(claims=[tech.Claim(field="core_chips", text="NPU-A 시제품 공개",
                                         evidence_id=0, quote="NPU-A 시제품을 공개했다")]),
            tech.Draft(claims=[tech.Claim(field="development_stage", text="2026년 3월 테이프아웃 완료",
                                         evidence_id=0, quote="2026년 3월 테이프아웃을 완료했다")]),
        ])
        def fake_llm(schema):
            if schema is tech.Draft:
                return Chain(next(drafts))
            if schema is tech.Relevance:
                return Chain(tech.Relevance(relevant_ids=[0], rewrite_query=""))
            return Chain(tech.Grounding(verdicts=[tech.Verdict(claim_id=0, supported=True)]))
        with (patch.object(tech, "search_many", side_effect=[[self.web], [stage]]) as search,
              patch.object(tech, "get_retriever") as retriever,
              patch.object(tech, "_llm", side_effect=fake_llm)):
            retriever.return_value.invoke.return_value = []
            result = tech.tech_summary_node({"company": {"name": "테스트칩"}})
        self.assertEqual(result["tech_summary"]["core_chips"], ["NPU-A 시제품 공개"])
        self.assertEqual(result["tech_summary"]["development_stage"], ["2026년 3월 테이프아웃 완료"])
        self.assertEqual(result["tech_summary"]["field_status"]["development_stage"], "supported")
        self.assertEqual(result["tech_summary"]["industry_context_status"], "no_reports")
        self.assertEqual(result["tech_summary"]["evidence"][1]["url"], stage["url"])
        self.assertTrue(any("테이프아웃" in q for q in search.call_args_list[1].args[0]))
        self.assertNotEqual(search.call_args_list[0].args[0], search.call_args_list[1].args[0])

    def test_web_search_failure_is_not_reported_as_public_information_missing(self):
        with (patch.object(tech, "search_many", side_effect=RuntimeError("Tavily quota")),
              patch.object(tech, "_llm") as llm):
            with self.assertRaisesRegex(RuntimeError, "Tavily quota"):
                tech.tech_summary_node({"company": {"name": "테스트칩"}})
        llm.assert_not_called()

    def test_missing_chroma_does_not_remove_verified_web_facts(self):
        web = {**self.web, "content": "테스트칩은 2026년 3월 테이프아웃을 완료했다."}
        claim = tech.Claim(field="development_stage", text="2026년 3월 테이프아웃 완료",
                           evidence_id=0, quote="2026년 3월 테이프아웃을 완료했다")
        with (patch.object(tech, "get_retriever", side_effect=FileNotFoundError("vectorstore 없음")),
              patch.object(tech, "search_many", return_value=[web]),
              patch.object(tech, "_llm") as llm):
            llm.side_effect = lambda schema: Chain({
                tech.Relevance: tech.Relevance(relevant_ids=[0], rewrite_query=""),
                tech.Draft: tech.Draft(claims=[claim]),
                tech.Grounding: tech.Grounding(verdicts=[tech.Verdict(claim_id=0, supported=True)]),
            }[schema])
            result = tech.tech_summary_node({"company": {"name": "테스트칩"}})
        summary = result["tech_summary"]
        self.assertEqual(summary["development_stage"], [claim.text])
        self.assertEqual(summary["industry_context_status"], "retrieval_unavailable")
        self.assertEqual(summary["industry_context"], [])
        self.assertIn("vectorstore", summary["diagnostics"]["rag"]["error"])
        self.assertEqual([s["kind"] for s in result["sources"]], ["web"])

    def test_context_requires_valid_web_fact_and_report_quote_and_grounding(self):
        item = {"kind": "web", "source": self.web, "title": self.web["title"], "text": self.web["content"]}
        claim = tech.Claim(field="core_chips", text="NPU-A 시제품 공개", evidence_id=0, quote="NPU-A 시제품을 공개했다")
        base = dict(topic="performance", text="업계 최상위 성능", report_id=0,
                    quote="메모리 대역폭의 영향을 받는다", comparison="comparable", company_fact_ids=[0])
        insights = [tech.IndustryInsight(**base),
                    tech.IndustryInsight(**{**base, "company_fact_ids": [99]}),
                    tech.IndustryInsight(**{**base, "quote": "없는 수치"})]
        with (patch.object(tech, "get_retriever") as retriever,
              patch.object(tech, "_llm") as llm):
            retriever.return_value.invoke.return_value = [self.report]
            llm.side_effect = lambda schema: Chain(
                tech.IndustryDraft(insights=insights) if schema is tech.IndustryDraft else
                tech.Grounding(verdicts=[tech.Verdict(claim_id=0, supported=False)]))
            context, sources, status, diagnostic = tech._industry_context("테스트칩", "엣지", [claim], [item])
        self.assertEqual((context, sources, status), ([], [], "no_supported_context"))
        self.assertEqual(diagnostic["invalid_insights"], 2)
        self.assertEqual(diagnostic["unsupported_insights"], 1)

    def test_long_web_content_keeps_quote_after_800_characters(self):
        web = {**self.web, "content": "설명 " * 400 + "테스트칩은 NPU-A 시제품을 공개했다."}
        with patch.object(tech, "search_many", return_value=[web]):
            items = tech._collect_web(["테스트칩"])
        self.assertIn("시제품을 공개했다", items[0]["text"])
        self.assertIn("시제품을 공개했다", tech._source(items[0], "테스트칩")["snippet"])

    def test_unquoted_or_unsupported_claims_are_removed(self):
        item = {"kind": "web", "source": self.web, "title": self.web["title"],
                "text": self.web["content"]}
        with patch.object(tech, "_llm") as llm:
            llm.side_effect = lambda schema: Chain({
                tech.Draft: tech.Draft(claims=[
                    tech.Claim(field="performance_metrics", text="100 TOPS",
                               evidence_id=0, quote="100 TOPS"),
                    tech.Claim(field="core_chips", text="NPU-A 양산 완료",
                               evidence_id=0, quote="NPU-A 시제품을 공개했다"),
                ]),
                tech.Grounding: tech.Grounding(
                    verdicts=[tech.Verdict(claim_id=0, supported=False)]),
            }[schema])
            audit = []
            self.assertEqual(tech._grounded("테스트칩", [item], audit), [])
            self.assertEqual([r["reason"] for r in audit],
                             ["quote_not_in_source", "grounding_not_supported"])

    def test_report_cannot_become_company_fact_even_when_company_is_mentioned(self):
        item = {"kind": "report", "source": self.report, "title": "테스트칩 일반 기술 보고서",
                "text": self.report.page_content}
        with patch.object(tech, "_llm") as llm:
            llm.return_value = Chain(tech.Draft(claims=[
                tech.Claim(field="strengths", text="메모리 대역폭이 강점",
                           evidence_id=0, quote="메모리 대역폭의 영향을 받는다"),
            ]))
            self.assertEqual(tech._grounded("테스트칩", [item]), [])

@unittest.skipUnless(os.getenv("RUN_KURE_INTEGRATION") == "1",
                     "RUN_KURE_INTEGRATION=1 일 때 실제 KURE-v1 모델을 로드한다")
class KureChromaIntegrationTests(unittest.TestCase):
    def test_real_embedding_and_chroma_search(self):
        from langchain_chroma import Chroma

        embedder = embeddings.LocalEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            store = Chroma.from_documents(
                [fixture_document(entry) for entry in FIXTURE["documents"]],
                embedder,
                collection_name="tech_summary_test",
                persist_directory=directory,
            )
            results = store.similarity_search(
                "가온-X1 7nm 테이프아웃", k=2, filter={"doc_type": "tech"})
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].metadata["source"], "company_brief.md")
            self.assertTrue(all(doc.metadata["doc_type"] == "tech" for doc in results))


if __name__ == "__main__":
    unittest.main()
