"""투자 판단 에이전트 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_judge.py -q"""
from itertools import product

import pytest

from agents import judge
from config import WEIGHTS

KEYS = list(WEIGHTS)
SRC = [{"company": "A", "node": "team", "title": "기사", "publisher": "언론", "date": "2026",
        "url": "http://x", "snippet": "s"}]
STATE = {"company": {"name": "A"}, "sources": SRC + [{"company": "B", "title": "다른 기업", "url": "http://b"}],
         "tech_summary": {"x": 1}, "market_analysis": {"y": 1}, "team_analysis": {"z": 1},
         "competitor_analysis": {"w": 1}}


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """모든 테스트에서 채점 로그를 임시 폴더로 돌린다 (상수 이름이 틀리면 raising 기본값으로 바로 실패)."""
    path = tmp_path / "judge_log.jsonl"
    monkeypatch.setattr(judge, "JUDGE_LOG_PATH", path)
    return path


def item(score, ids=(0,), ins=False):
    return judge.ItemScore(score=score, evidence="근거", source_ids=list(ids), insufficient=ins)


def card(legal=False, legal_ids=(), **over):
    base = {k: item(4) for k in KEYS}
    base.update(over)
    return judge.Scorecard(**base,
                           legal_risk=judge.LegalRisk(unresolved=legal, evidence="소송", source_ids=list(legal_ids)),
                           risks=[judge.Risk(category="시장", description="r")])


def evaluate(c, state=STATE):
    items = judge.apply_evidence_rules(c, state, judge.company_sources(state))
    total = judge.total_score(items)
    return items, total, judge.decide(items, total, False)


def run_node(monkeypatch, c, state=STATE):
    class Fake:
        def __init__(self, *a, **k): pass
        def with_structured_output(self, _): return self
        def with_retry(self, **_): return self
        def invoke(self, prompt): return c
    monkeypatch.setattr(judge, "ChatOpenAI", Fake)
    return judge.judge_node(state)


def test_all_4_invests():
    _, total, (decision, why) = evaluate(card())
    assert total == 80.0 and decision == "투자" and why == []


def test_cutoff_boundary():
    at70 = dict(tech=item(3), market=item(4), team=item(4), traction=item(3), competition=item(3), deal=item(2))
    assert evaluate(card(**at70))[1:] == (70.0, ("투자", []))
    assert evaluate(card(**{**at70, "deal": item(1)}))[1:] == (69.0, ("보류", ["환산 점수 69점 < 기준 70점"]))


def test_core_item_hold_even_when_total_high():
    _, total, (decision, why) = evaluate(card(**{**{k: item(5) for k in KEYS}, "team": item(1)}))
    assert total > 70 and decision == "보류" and any("창업자·팀" in w for w in why)


def test_core_item_2_points_is_not_core_shortfall():
    """핵심 역량 미달은 1점 이하 (2점은 미달이 아님)."""
    _, total, (decision, why) = evaluate(card(**{**{k: item(5) for k in KEYS}, "tech": item(2)}))
    assert total > 70 and decision == "투자" and why == []


def test_no_source_caps_at_2():
    items, _, _ = evaluate(card(tech=item(5, ids=())))
    assert items["tech"]["score"] == 2 and items["tech"]["insufficient"]
    assert items["tech"]["evidence"].startswith("정보 부족")


def test_invalid_source_index_caps_at_2():
    assert evaluate(card(team=item(5, ids=(7,))))[0]["team"]["score"] == 2


def test_llm_flag_insufficient_caps_at_2():
    assert evaluate(card(deal=item(5, ins=True)))[0]["deal"]["score"] == 2


def test_upstream_todo_caps_tech_and_traction_even_if_llm_cites_source():
    state = {**STATE, "tech_summary": {"todo": True}}
    items, _, (decision, _) = evaluate(card(tech=item(4), traction=item(4)), state)
    assert items["tech"]["score"] == 2 and items["traction"]["score"] == 2 and decision == "보류"
    assert items["market"]["score"] == 4      # 다른 항목은 영향 없음


@pytest.mark.parametrize("key,state_key,key_in_card", [
    ("team", "team_analysis", "team"), ("competition", "competitor_analysis", "competition"),
    ("market", "market_analysis", "market")])
def test_missing_upstream_caps(key, state_key, key_in_card):
    state = {**STATE, state_key: None}
    assert evaluate(card(**{key_in_card: item(5)}), state)[0][key]["score"] == 2


def test_market_info_insufficient_flag():
    state = {**STATE, "market_analysis": {"info_insufficient": True}}
    assert evaluate(card(market=item(5)), state)[0]["market"]["score"] == 2


def test_score_out_of_range_is_clamped_not_crash():
    items, _, _ = evaluate(card(tech=item(9), deal=item(0)))
    assert items["tech"]["score"] == 5 and items["deal"]["score"] == 1


def test_total_matches_reporter_for_all_combinations():
    from agents.reporter import recompute_total
    for combo in product(range(1, 6), repeat=len(KEYS)):
        it = {k: {"score": s} for k, s in zip(KEYS, combo)}
        assert judge.total_score(it) == recompute_total({"items": it}), combo


def test_node_rejected_format_and_company_filter(monkeypatch):
    out = run_node(monkeypatch, card(tech=item(1)))
    scores, rej = out["scores"], out["rejected"][0]
    assert scores["decision"] == "보류"
    assert rej == {"company": "A", "reason": "; ".join(scores["hold_reasons"]), "total": scores["total"]}
    assert all(s["title"] != "다른 기업" for it in scores["items"].values() for s in it["sources"])


def test_node_invest_has_no_rejected(monkeypatch):
    out = run_node(monkeypatch, card())
    assert out["scores"]["decision"] == "투자" and "rejected" not in out


def test_node_legal_risk_needs_source(monkeypatch):
    with_src = run_node(monkeypatch, card(legal=True, legal_ids=(0,)))["scores"]
    assert with_src["decision"] == "보류" and with_src["legal_risk"]["unresolved"]
    no_src = run_node(monkeypatch, card(legal=True, legal_ids=()))["scores"]
    assert no_src["decision"] == "투자" and not no_src["legal_risk"]["unresolved"]


def test_reporter_verifier_accept_output(monkeypatch):
    from agents.reporter import build_score_table, find_insufficient_items
    from agents.verifier import check_total_score
    scores = run_node(monkeypatch, card(team=item(5, ins=True)))["scores"]
    assert check_total_score(scores) == []
    assert "환산 점수" in build_score_table(scores)
    assert find_insufficient_items(scores) == ["창업자·팀"]


def test_log_has_raw_and_final_scores(monkeypatch, isolated_log):
    import json
    run_node(monkeypatch, card(tech=item(5, ids=()), deal=item(4)))
    lines = isolated_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["company"] == "A" and rec["decision"] and set(rec["items"]) == set(KEYS)
    assert all({"raw_score", "score", "insufficient", "evidence"} <= set(v) for v in rec["items"].values())
    assert rec["items"]["tech"]["raw_score"] == 5 and rec["items"]["tech"]["score"] == 2   # 2점 상한
    assert rec["items"]["deal"]["raw_score"] == rec["items"]["deal"]["score"] == 4
    assert len(rec["items"]["tech"]["evidence"]) <= 80


def test_log_records_legal_risk_judgement(monkeypatch, isolated_log):
    import json
    run_node(monkeypatch, card(legal=True, legal_ids=()))    # LLM은 미해소로 판정했지만 출처가 없어 무시되는 경우
    legal = json.loads(isolated_log.read_text(encoding="utf-8"))["legal_risk"]
    assert legal["raw_unresolved"] is True and legal["unresolved"] is False
    assert legal["evidence"] == "소송" and legal["sources"] == []


def test_log_write_failure_does_not_stop_node(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(judge, "JUDGE_LOG_PATH", blocker / "sub" / "log.jsonl")   # 파일 아래 경로: 쓸 수 없음
    out = run_node(monkeypatch, card())
    assert out["scores"]["decision"] == "투자"


def test_cautions_for_low_core_items():
    # 기술: 정보 부족(출처 없음)으로 2점 제한, 팀: 실제로 2점으로 평가. 투자 결정과는 별개
    items, _, _ = evaluate(card(**{**{k: item(5) for k in KEYS}, "tech": item(5, ids=()), "team": item(2)}))
    cautions = judge.build_cautions(items)
    by_item = {c["item"]: c for c in cautions}
    assert set(by_item) == {"tech", "team"}
    assert by_item["tech"]["insufficient"] and "정보 부족" in by_item["tech"]["comment"]
    assert not by_item["team"]["insufficient"] and "낮은 평가" in by_item["team"]["comment"]
    assert all(c["comment"].startswith(c["label"]) and str(c["score"]) in c["comment"] for c in cautions)


def test_no_caution_when_core_items_are_3_or_more_or_non_core_low():
    items, _, _ = evaluate(card(**{**{k: item(4) for k in KEYS}, "tech": item(3), "deal": item(1), "traction": item(2)}))
    assert judge.build_cautions(items) == []     # 핵심 항목(팀·기술)이 아닌 낮은 점수는 코멘트 대상 아님


def test_node_returns_cautions_and_logs_them(monkeypatch, isolated_log):
    import json
    out = run_node(monkeypatch, card(**{**{k: item(5) for k in KEYS}, "tech": item(2)}))
    scores = out["scores"]
    assert scores["decision"] == "투자" and [c["item"] for c in scores["cautions"]] == ["tech"]
    logged = json.loads(isolated_log.read_text(encoding="utf-8"))
    assert logged["cautions"] == [c["comment"] for c in scores["cautions"]]


# ── 평가 기록과 비교한 재채점 ─────────────────────────────────
def run_node_by_seed(monkeypatch, cards_by_seed, state=STATE):
    """seed마다 다른 채점 결과를 돌려주는 가짜 LLM으로 노드를 실행한다."""
    calls = []

    class Fake:
        def __init__(self, *a, seed=None, **k): self.seed = seed
        def with_structured_output(self, _): return self
        def with_retry(self, **_): return self
        def invoke(self, prompt):
            calls.append(self.seed)
            return cards_by_seed[self.seed]
    monkeypatch.setattr(judge, "ChatOpenAI", Fake)
    return judge.judge_node(state), calls


def write_history(path, totals, decision, version):
    import json
    with path.open("w", encoding="utf-8") as f:
        for t in totals:
            f.write(json.dumps({"company": "A", "total": t, "decision": decision, "eval_version": version}) + "\n")


def test_no_history_scores_once(monkeypatch, isolated_log):
    import json
    out, calls = run_node_by_seed(monkeypatch, {judge.LLM_SEED: card()})
    assert calls == [judge.LLM_SEED] and out["scores"]["decision"] == "투자"
    rec = json.loads(isolated_log.read_text(encoding="utf-8"))
    assert rec["eval_version"] == judge.eval_version() and rec["consistency"] == {"history": 0, "rescored": False}


def test_inconsistent_with_history_rescores_with_median(monkeypatch, isolated_log):
    import json
    write_history(isolated_log, [60, 62], "보류", judge.eval_version())      # 같은 버전의 과거 기록: 보류
    first = card()                                                          # 처음 채점: 전 항목 4점 → 80점 투자
    low = card(**{k: item(3) for k in KEYS})                                # 재채점 2회: 전 항목 3점
    out, calls = run_node_by_seed(monkeypatch, {judge.LLM_SEED: first, judge.LLM_SEED + 1: low, judge.LLM_SEED + 2: low})
    assert calls == [judge.LLM_SEED, judge.LLM_SEED + 1, judge.LLM_SEED + 2]
    assert out["scores"]["total"] == 60.0 and out["scores"]["decision"] == "보류"   # 항목별 중앙값 3점
    rec = json.loads(isolated_log.read_text(encoding="utf-8").splitlines()[-1])
    c = rec["consistency"]
    assert c["rescored"] and c["first_total"] == 80.0 and c["history"] == 2 and not c["still_inconsistent"]


def test_history_from_other_version_is_ignored(monkeypatch, isolated_log):
    write_history(isolated_log, [40, 42], "보류", "old-version")            # 평가 로직이 바뀌기 전 기록
    out, calls = run_node_by_seed(monkeypatch, {judge.LLM_SEED: card()})
    assert calls == [judge.LLM_SEED] and out["scores"]["decision"] == "투자"


def test_median_uses_final_scores_after_evidence_rules():
    """출처가 없어 2점으로 깎일 5점이 중앙값으로 뽑히지 않는다 (최종 점수 2·4·1의 중앙값은 2)."""
    samples = [card(market=item(5, ids=())), card(market=item(4)), card(market=item(1))]
    sources = judge.company_sources(STATE)
    merged = judge.median_card(samples, STATE, sources)
    assert judge.score_card(merged, STATE, sources)["items"]["market"]["score"] == 2


def test_median_legal_risk_follows_effective_majority():
    """출처 규칙을 적용한 실제 판정(미해소·해소·해소)의 다수인 '해소'를 따른다."""
    samples = [card(legal=True, legal_ids=(0,)), card(legal=True, legal_ids=()), card(legal=False)]
    sources = judge.company_sources(STATE)
    merged = judge.median_card(samples, STATE, sources)
    assert judge.score_card(merged, STATE, sources)["legal_risk"]["unresolved"] is False


def test_rescore_failure_keeps_first_result(monkeypatch, isolated_log):
    import json
    write_history(isolated_log, [60, 62], "보류", judge.eval_version())
    calls = []

    class Fake:
        def __init__(self, *a, seed=None, **k): self.seed = seed
        def with_structured_output(self, _): return self
        def with_retry(self, **_): return self
        def invoke(self, prompt):
            calls.append(self.seed)
            if self.seed != judge.LLM_SEED:
                raise TimeoutError("rate limit")
            return card()
    monkeypatch.setattr(judge, "ChatOpenAI", Fake)
    out = judge.judge_node(STATE)
    assert out["scores"]["total"] == 80.0
    c = json.loads(isolated_log.read_text(encoding="utf-8").splitlines()[-1])["consistency"]
    assert not c["rescored"] and "rate limit" in c["rescore_error"]


def test_history_skips_broken_lines(isolated_log):
    import json
    good = json.dumps({"company": "A", "total": 70, "decision": "투자", "eval_version": "v"}, ensure_ascii=False)
    isolated_log.write_bytes((good + "\n[1, 2]\n" + good[:-5]).encode("utf-8") + "평가".encode("utf-8")[:-1])
    assert len(judge.past_records("A", "v")) == 1
