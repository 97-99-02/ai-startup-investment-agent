"""웹 + RAG 결과의 판단·보고서·검증 연결 테스트 (API 호출 없음)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import judge, reporter, verifier
from config import WEIGHTS


class HybridSummaryFlowTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "company": {"name": "테스트칩"},
            "tech_summary": {
                "development_stage": ["2026년 테이프아웃 완료"],
                "performance_metrics": ["24 TOPS, 8 W"],
                "info_insufficient": True, "missing_fields": ["weaknesses"],
                "industry_context": [{
                    "topic": "performance", "text": "비교 조건 부족", "comparison": "conditions_missing",
                    "report_source_id": "report:file2024:p987:abc",
                    "company_source_ids": ["web:https://example.test/12345"],
                    "page": 987, "date": "2024", "quote": "추적용 원문 888 TOPS",
                }],
                "diagnostics": {"retrieved_web": 777,
                                "rejected_claims": [{"text": "999 TOPS", "reason": "quote_not_in_source"}]},
            },
            "market_analysis": {"ready": True}, "team_analysis": {"ready": True},
            "competitor_analysis": {"ready": True}, "candidates": ["테스트칩"],
            "sources": [{"company": "테스트칩", "node": "tech_summary", "kind": "web",
                         "title": "테이프아웃 완료", "url": "https://example.test/chip",
                         "snippet": "테스트칩은 2026년 테이프아웃을 완료했다."}],
        }

    def test_judge_receives_both_roles_without_rejected_draft(self):
        prompts = []
        card = judge.Scorecard(
            **{k: judge.ItemScore(score=3 if k == "tech" else 4, evidence="테이프아웃 완료",
                                 source_ids=[0], insufficient=False) for k in WEIGHTS},
            legal_risk=judge.LegalRisk(unresolved=False, evidence="없음", source_ids=[]), risks=[])
        class FakeLLM:
            def __init__(self, **_kwargs): pass
            def with_structured_output(self, _schema): return self
            def with_retry(self, **_): return self
            def invoke(self, prompt):
                prompts.append(prompt)
                return card
        original = copy.deepcopy(self.state)
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(judge, "ChatOpenAI", FakeLLM),
                  patch.object(judge, "JUDGE_LOG_PATH", Path(directory) / "judge.jsonl")):
                result = judge.judge_node(self.state)
        self.assertNotIn("999 TOPS", prompts[0])
        self.assertNotIn('"diagnostics"', prompts[0])
        self.assertIn("24 TOPS", prompts[0])
        self.assertIn('"industry_context"', prompts[0])
        self.assertEqual(result["scores"]["items"]["tech"]["score"], 3)
        self.assertFalse(result["scores"]["items"]["tech"]["insufficient"])
        self.assertEqual(self.state, original)

    def test_report_materials_keep_facts_and_context_without_diagnostics(self):
        state = {**self.state, "scores": {"decision": "투자"}}
        materials = reporter.build_materials(state, [{"n": 1, "label": "웹페이지", "title": "테이프아웃"}])
        self.assertNotIn("diagnostics", materials["tech_summary"])
        self.assertEqual(materials["tech_summary"]["industry_context"], state["tech_summary"]["industry_context"])
        self.assertEqual(materials["tech_summary"]["development_stage"], ["2026년 테이프아웃 완료"])
        self.assertIn("diagnostics", state["tech_summary"])

    def test_verifier_excludes_trace_numbers_and_rejected_claims(self):
        evidence = verifier._evidence_json(self.state)
        for metadata in ("999 TOPS", "888 TOPS", "12345", "987", "777"):
            self.assertNotIn(metadata, evidence)
        self.assertIn("24 TOPS", evidence)
        self.assertIn("비교 조건 부족", evidence)
        clean = json.loads(evidence)["tech_summary"]
        self.assertEqual(set(clean["industry_context"][0]), {"topic", "text", "comparison"})
        self.assertIn("diagnostics", self.state["tech_summary"])
