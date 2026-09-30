"""시장성 평가 에이전트 개선 전후 비교 (docs/시장성평가-개선방안.md 6번)

정답지 gold.json: 세부 분야마다 문서에 실제로 있는 시장 규모·성장률 수치와 그 수치가 든 청크 id.
  primary=true 는 반드시 뽑아야 하는 대표 수치, false 는 뽑혀도 오답이 아닌 해당 분야 수치.

[검색 단계] LLM 없이 몇 초. 결과가 매번 같아 1회면 된다
  - hit@4, mrr@10: 분야 질의(SEGMENT_QUERIES) 하나하나로 검색했을 때 정답 청크가 몇 위에 나오나
    (정답 청크가 있는 문서 유형 안에서만 잰다. 설계 2.4 필터와 같다)
  - 수치 도달률: market_node가 LLM에 넘기는 조각 묶음에 대표 수치(primary)의 청크가 들어갔나
    market만 검색했을 때와 tech 재검색까지 했을 때를 따로 잰다
  - 분할 수치 도달률: 표 제목·단위와 값이 다른 청크로 갈라진 수치(context_ids 있음)의 조각이 전부 들어갔나
[노드 단계] market_node를 분야마다 실행. LLM 출력이 흔들려 --runs 로 여러 번 돌려 평균
  - 충분성: info_insufficient가 False인 비율
  - 수치 재현율: 대표 수치 중 segment_specific=True로 맞게 뽑힌 비율
  - 수치 정밀도: segment_specific=True로 뽑힌 수치 중 정답(대표+인정)과 맞는 비율
  - 분야 오판정: 대표 수치를 뽑긴 했는데 segment_specific=False로 떨어진 개수
  - LLM 호출(=searched 경로 수), 소요 시간

수치 일치 기준: kind가 같고, 값의 숫자들이 같고(쉼표·공백 무시), 연도가 둘 다 있으면 연도도 같다('24 = 2024).

실행 (프로젝트 루트에서):
  uv run python -m eval.market.evaluate --label baseline                 # 검색 + 노드 1회
  uv run python -m eval.market.evaluate --label baseline --runs 3        # 노드 3회 평균
  uv run python -m eval.market.evaluate --label baseline --mode retrieval
결과: eval/market/results/<label>.json (분야별 상세 포함), 요약 한 줄은 results/summary.md에 누적
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agents import market
from rag.retriever import get_retriever

load_dotenv()
HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "gold.json"
RESULT_DIR = HERE / "results"
HIT_K, MRR_K = 4, 10


def nums(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", re.sub(r"[\s,]", "", text or ""))


def years(text: str) -> list[int]:
    return [int(y) + 2000 if int(y) < 100 else int(y) for y in nums(text)]  # '24 → 2024


def same_figure(f: dict, g: dict) -> bool:
    if f["kind"] != g["kind"] or nums(f["value"]) != nums(g["value"]):
        return False
    return not (f.get("year") and g.get("year")) or years(f["year"]) == years(g["year"])


def load_gold() -> dict:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["segments"]
    return {seg: v for seg, v in gold.items() if v.get("figures")}  # 정답을 아직 안 채운 분야는 건너뛴다


# ---------- 검색 단계 ----------
def eval_retrieval(gold: dict) -> dict:
    per_segment, all_hits, all_rr = {}, [], []
    for seg, g in gold.items():
        queries = market.SEGMENT_QUERIES.get(seg, market.SEGMENT_QUERIES["기타"])
        primary = [f for f in g["figures"] if f["primary"]]
        split = [f for f in g["figures"] if f.get("context_ids")]
        gold_ids = {cid for f in primary for cid in f["chunk_ids"]}
        gold_types = {f["doc_type"] for f in primary}

        hits, rrs = [], []
        for doc_type in sorted(gold_types):
            for q in queries:
                ranked = [d.metadata["chunk_id"] for d in get_retriever(doc_type, k=MRR_K).invoke(q)]
                rank = next((i + 1 for i, cid in enumerate(ranked) if cid in gold_ids), None)
                hits.append(rank is not None and rank <= HIT_K)
                rrs.append(1 / rank if rank else 0.0)

        market_docs = market.retrieve(queries, "market")
        full_docs = market_docs + market.retrieve(queries, "tech")

        def reach(docs, figs):
            if not figs:
                return None
            ids = {d.metadata["chunk_id"] for d in docs}
            ok = [any(c in ids for c in f["chunk_ids"]) and all(c in ids for c in f.get("context_ids", []))
                  for f in figs]
            return round(sum(ok) / len(ok), 3)

        per_segment[seg] = {
            f"hit@{HIT_K}": round(sum(hits) / len(hits), 3),
            f"mrr@{MRR_K}": round(sum(rrs) / len(rrs), 3),
            "reach_market": reach(market_docs, primary),
            "reach_full": reach(full_docs, primary),
            "reach_split_full": reach(full_docs, split),
            "chunks_to_llm_full": len(full_docs),
        }
        all_hits += hits
        all_rr += rrs

    def avg(key):
        vals = [v[key] for v in per_segment.values() if v[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "summary": {
            f"hit@{HIT_K}": round(sum(all_hits) / len(all_hits), 3),
            f"mrr@{MRR_K}": round(sum(all_rr) / len(all_rr), 3),
            "reach_market": avg("reach_market"),
            "reach_full": avg("reach_full"),
            "reach_split_full": avg("reach_split_full"),
        },
        "per_segment": per_segment,
    }


# ---------- 노드 단계 ----------
def score_node(analysis: dict, g: dict) -> dict:
    figures = analysis["figures"]
    specific = [f for f in figures if f["segment_specific"]]
    primary = [gf for gf in g["figures"] if gf["primary"]]
    found = [any(same_figure(f, gf) for f in specific) for gf in primary]
    misjudged = sum(
        not ok and any(same_figure(f, gf) for f in figures if not f["segment_specific"])
        for ok, gf in zip(found, primary)
    )
    correct = [any(same_figure(f, gf) for gf in g["figures"]) for f in specific]
    return {
        "sufficient": not analysis["info_insufficient"],
        "recall": round(sum(found) / len(found), 3),
        "precision": round(sum(correct) / len(correct), 3) if correct else None,
        "misjudged": misjudged,
        "n_specific": len(specific),
        "n_figures": len(figures),
        "n_drivers": len(analysis["demand_drivers"]),
        "n_risks": len(analysis["market_risks"]),
        "llm_calls": len(analysis["searched"]),
        "searched": analysis["searched"],
    }


def eval_node(gold: dict, runs: int) -> dict:
    cases = []
    for run in range(runs):
        for seg, g in gold.items():
            t0 = time.time()
            out = market.market_node({"company": g["company"]})
            s = score_node(out["market_analysis"], g)
            s.update(run=run, segment=seg, sec=round(time.time() - t0, 1), n_sources=len(out["sources"]),
                     figures=out["market_analysis"]["figures"])
            cases.append(s)
            print(f"  [{run + 1}/{runs}] {seg}: 충분={s['sufficient']} 재현율={s['recall']} "
                  f"정밀도={s['precision']} 오판정={s['misjudged']} 경로={s['searched']} {s['sec']}s")

    def mean(key):
        vals = [c[key] for c in cases if c[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "summary": {
            "sufficient_rate": mean("sufficient"),
            "recall": mean("recall"),
            "precision": mean("precision"),
            "misjudged": mean("misjudged"),
            "llm_calls": mean("llm_calls"),
            "sec": mean("sec"),
        },
        "cases": cases,
    }


def append_summary(label: str, result: dict):
    path = RESULT_DIR / "summary.md"
    header = ("| 시각 | 변형 | hit@4 | mrr@10 | 도달(market) | 도달(full) | 분할 도달 | 충분성 | 재현율 | 정밀도 | 오판정 | LLM 호출 | 초 |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    r = (result.get("retrieval") or {}).get("summary", {})
    n = (result.get("node") or {}).get("summary", {})
    cells = [datetime.now().strftime("%m-%d %H:%M"), label,
             r.get(f"hit@{HIT_K}"), r.get(f"mrr@{MRR_K}"), r.get("reach_market"), r.get("reach_full"),
             r.get("reach_split_full"), n.get("sufficient_rate"), n.get("recall"), n.get("precision"),
             n.get("misjudged"), n.get("llm_calls"), n.get("sec")]
    row = "| " + " | ".join("-" if c is None else str(c) for c in cells) + " |\n"
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as fp:
        fp.write(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="변형 이름. 예: baseline, +query_transform")
    ap.add_argument("--mode", choices=["all", "retrieval", "node"], default="all")
    ap.add_argument("--runs", type=int, default=1, help="노드 단계 반복 횟수")
    args = ap.parse_args()

    gold = load_gold()
    if not gold:
        raise SystemExit("gold.json에 figures가 채워진 분야가 없습니다.")
    print(f"평가 분야: {list(gold)}")

    result = {"label": args.label, "time": datetime.now().isoformat(timespec="seconds"),
              "segments": list(gold), "runs": args.runs}
    if args.mode in ("all", "retrieval"):
        result["retrieval"] = eval_retrieval(gold)
        print("검색:", json.dumps(result["retrieval"]["summary"], ensure_ascii=False))
    if args.mode in ("all", "node"):
        result["node"] = eval_node(gold, args.runs)
        print("노드:", json.dumps(result["node"]["summary"], ensure_ascii=False))

    RESULT_DIR.mkdir(exist_ok=True)
    path = RESULT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_summary(args.label, result)
    print(f"저장: {path}, 요약: {RESULT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
