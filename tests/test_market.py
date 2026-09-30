"""시장성 평가 에이전트 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_market.py -q"""
import pytest

from agents import market

# 실제 청크(KDB·수출입은행 요약)에서 따온 원문. 세 시장 수치가 한 조각에 함께 있다
SUMMARY_CHUNK = """AI반도체 시장은 2024년 1,830억 달러에서 2030년 1조 달러로 연평균 33% 성장할 전망
데이터센터용 AI반도체 시장은 2024년 1,240억 달러에서 2030년 8,600억 달러로 연평균 38% 성장할 전망
온디바이스용 AI반도체 시장은 2024년 592억 달러에서 2030년 1,430억 달러로 연평균 16% 성장할 전망"""


@pytest.mark.parametrize("kind, value, ok", [
    ("시장규모", "1,240억 달러", True),
    ("시장규모", "14백만달러", True),
    ("시장규모", "3조원", True),
    ("시장규모", "$12B", True),
    ("시장규모", "155", False),        # 차트 눈금값
    ("시장규모", "71억", False),       # 통화 단위 없음
    ("성장률", "38%", True),
    ("성장률", "38", False),
])
def test_has_unit(kind, value, ok):
    assert market.has_unit(kind, value) is ok


@pytest.mark.parametrize("scope, ok", [
    ("AI반도체 시장", True),
    ("AI 반도체", True),
    ("데이터센터용 AI반도체 시장", False),   # 'ai반도체'를 포함해도 다른 분야 수치
    ("온디바이스용 AI반도체 시장", False),
    ("기타", False),                         # 차트 범례의 '기타' 항목
])
def test_segment_specific_etc_excludes_other_segments(scope, ok):
    assert market.is_segment_specific(scope, SUMMARY_CHUNK, "기타") is ok


def test_segment_specific_needs_term_in_scope_and_body():
    assert market.is_segment_specific("데이터센터용 AI반도체 시장", SUMMARY_CHUNK, "데이터센터")
    assert market.is_segment_specific("온디바이스용 AI반도체 시장", SUMMARY_CHUNK, "엣지")
    assert not market.is_segment_specific("데이터센터용 AI반도체 시장", SUMMARY_CHUNK, "차량용")
    # scope에만 있고 원문에 없는 용어로는 인정하지 않는다 (LLM이 scope를 대상 분야로 바꿔 적는 경우)
    assert not market.is_segment_specific("자율주행차용 AI반도체", SUMMARY_CHUNK, "차량용")


def test_prompt_uses_label_for_etc(monkeypatch):
    """'기타'는 프롬프트에 'AI 반도체 전체'로 들어간다."""
    captured = {}

    class FakeLLM:
        def with_structured_output(self, _):
            return self

        def invoke(self, prompt):
            captured["prompt"] = prompt
            return market.MarketAnalysis(figures=[], demand_drivers=[], market_risks=[], summary="", evidence_ids=[])

    monkeypatch.setattr(market, "ChatOpenAI", lambda **_: FakeLLM())
    market.extract("엑시나", "기타", [])
    assert "(AI 반도체 전체)" in captured["prompt"] and "(기타)" not in captured["prompt"]


def test_extract_drops_unitless_and_unsourced(monkeypatch):
    from langchain_core.documents import Document

    figs = [
        market.MarketFigure(kind="시장규모", value="592억 달러", year="2024", scope="온디바이스용 AI반도체 시장", evidence_id=0),
        market.MarketFigure(kind="시장규모", value="155", year="2022", scope="PC용 AI 반도체", evidence_id=0),
        market.MarketFigure(kind="성장률", value="99%", year="", scope="온디바이스용 AI반도체", evidence_id=0),  # 원문에 없음
        market.MarketFigure(kind="성장률", value="16%", year="", scope="온디바이스용 AI반도체", evidence_id=5),  # 없는 조각 번호
    ]

    class FakeLLM:
        def with_structured_output(self, _):
            return self

        def invoke(self, _):
            return market.MarketAnalysis(figures=figs, demand_drivers=[], market_risks=[], summary="", evidence_ids=[0])

    monkeypatch.setattr(market, "ChatOpenAI", lambda **_: FakeLLM())
    _, out = market.extract("모빌린트", "엣지", [Document(page_content=SUMMARY_CHUNK + "\n155", metadata={})])
    assert [f["value"] for f in out] == ["592억 달러"]
    assert out[0]["segment_specific"]
