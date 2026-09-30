"""기술 요약 Self-RAG와 KURE-v1 검색 테스트.

기본 테스트: uv run python -m unittest discover -s tests -v
실제 임베딩/Chroma 테스트: RUN_KURE_INTEGRATION=1 uv run python -m unittest discover -s tests -v
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

    def test_company_fact_has_source_and_generic_report_is_excluded(self):
        with (
            patch.object(tech, "get_retriever") as retriever,
            patch.object(tech, "search_many", return_value=[self.web]),
            patch.object(tech, "_llm") as llm,
        ):
            retriever.return_value.invoke.return_value = [self.report]
            llm.side_effect = lambda schema: Chain({
                tech.Relevance: tech.Relevance(relevant_ids=[0, 1], rewrite_query=""),
                tech.Draft: tech.Draft(claims=[
                    tech.Claim(field="core_chips", text="NPU-A 시제품 공개",
                               evidence_id=0, quote="NPU-A 시제품을 공개했다"),
                ]),
                tech.Grounding: tech.Grounding(
                    verdicts=[tech.Verdict(claim_id=0, supported=True)]),
            }[schema])
            result = tech.tech_summary_node({"company": {"name": "테스트칩"}})

        self.assertEqual(result["tech_summary"]["core_chips"], ["NPU-A 시제품 공개"])
        self.assertIn("process", result["tech_summary"]["missing_fields"])
        self.assertEqual([source["url"] for source in result["sources"]],
                         ["https://example.org/chip"])
        self.assertEqual(result["sources"][0]["node"], "tech_summary")
        self.assertEqual(result["tech_summary"]["evidence"][0]["source_id"],
                         result["sources"][0]["source_id"])
        retriever.assert_called_once_with("tech", k=5)

    def test_dummy_data_covers_all_summary_fields(self):
        docs = [fixture_document(entry) for entry in FIXTURE["documents"]
                if entry["metadata"]["doc_type"] == "tech"]
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
        with (
            patch.object(tech, "get_retriever") as retriever,
            patch.object(tech, "search_many",
                         return_value=[fixture_web_result(entry) for entry in FIXTURE["web_results"]]),
            patch.object(tech, "_llm") as llm,
        ):
            retriever.return_value.invoke.return_value = docs
            llm.side_effect = lambda schema: Chain({
                tech.Relevance: tech.Relevance(relevant_ids=[0, 1, 2, 3], rewrite_query=""),
                tech.Draft: tech.Draft(claims=claims),
                tech.Grounding: tech.Grounding(verdicts=[
                    tech.Verdict(claim_id=i, supported=True) for i in range(len(claims))
                ]),
            }[schema])
            result = tech.tech_summary_node({"company": FIXTURE["company"]})

        for field, value in FIXTURE["expected"].items():
            self.assertEqual(result["tech_summary"][field], [value])
        self.assertFalse(result["tech_summary"]["info_insufficient"])
        self.assertEqual(result["tech_summary"]["missing_fields"], [])
        self.assertEqual(len(result["sources"]), 3)
        self.assertNotIn("일반 기술 보고서", [source["title"] for source in result["sources"]])
        source_ids = {source["source_id"] for source in result["sources"]}
        self.assertTrue(all(trace["source_id"] in source_ids
                            for trace in result["tech_summary"]["evidence"]))
        self.assertTrue(result["sources"][0]["source_id"].startswith("report:company_brief.md:p1:"))
        self.assertNotIn("chunk_id", result["sources"][0])
        self.assertNotIn("chunk_id", result["tech_summary"]["evidence"][0])
        self.assertEqual(result["tech_summary"]["evidence"][0]["source_file"],
                         "company_brief.md")
        self.assertTrue(all((PROJECT_ROOT / source["source_path"]).is_file()
                            for source in result["sources"]))
        self.assertEqual(result["tech_summary"]["evidence"][3]["url"],
                         "https://example.org/fictional/performance")
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
        lines = [
            f"# {FIXTURE['company']['name']} 기술 요약 (더미 테스트)",
            "",
            "> 가상 자료와 모의 LLM 응답으로 만든 테스트 결과입니다. 실제 기업 분석 결과가 아닙니다.",
        ]
        for field, label in labels.items():
            lines.extend(["", f"## {label}", *[f"- {value}" for value in result["tech_summary"][field]]])
            for trace in result["tech_summary"]["evidence"]:
                if trace["field"] == field:
                    page = f" p.{trace['page']}" if trace.get("page") is not None else ""
                    lines.append(f"  - 근거: {trace['title']}{page} · \"{trace['quote']}\"")
        lines.extend(["", "## 사용한 출처"])
        for source in result["sources"]:
            local_file = (PROJECT_ROOT / source["source_path"]).resolve()
            lines.append(
                f"- [{source['title']}]({local_file}) "
                f"({source['kind']}, 원본 파일)"
            )
        markdown = "\n".join(lines) + "\n"
        output.with_suffix(".md").write_text(markdown, encoding="utf-8")
        print(f"\n{markdown}결과 파일: {output.with_suffix('.md')}\n", flush=True)

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
            self.assertEqual(tech._grounded("테스트칩", [item]), [])

    def test_generic_report_cannot_become_company_fact(self):
        item = {"kind": "report", "source": self.report, "title": "일반 기술 보고서",
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
