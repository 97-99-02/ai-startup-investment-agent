"""시장성 평가 정답지(gold.json) 작성을 돕는 후보 청크 추출 (docs/시장성평가-개선방안.md 6번)

1) rag.ingest.load_documents()로 실제 적재와 같은 청크를 만든다 (chunk_id도 적재 순서와 같다)
2) 금액(억·조·백만 달러/원) 또는 %가 있고, '시장·규모·성장률·CAGR' 같은 말이 함께 나오는 청크를 고른다
3) 세부 분야 용어(agents.market.SEGMENT_TERMS)가 들어 있으면 그 분야 후보로 묶는다
4) candidates.md에 사람이 읽기 좋게 쓴다. 이걸 보고 gold.json의 figures와 chunk_ids를 채운다

vectorstore를 읽지 않으므로 적재 중에도 실행할 수 있다. 문서나 청크 설정이 바뀌면 다시 실행한다.
실행 (프로젝트 루트에서): uv run python -m eval.market.find_candidates
"""
import re
from pathlib import Path

from agents.market import SEGMENT_TERMS, _compact
from rag.ingest import load_documents

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "candidates.md"

MONEY = re.compile(r"\d[\d,.]*\s*(?:억|조|백만|십억)?\s*(?:달러|원|USD|\$)|\$\s*\d[\d,.]*\s*(?:B|bn|billion|억)?")
PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
MARKET_WORDS = ("시장", "규모", "성장률", "cagr", "연평균", "전망")


def numbers_in(text: str) -> list[str]:
    return [m.group(0).strip() for m in MONEY.finditer(text)] + [m.group(0).strip() for m in PERCENT.finditer(text)]


def segments_of(text: str) -> list[str]:
    body = _compact(text).lower()
    return [seg for seg, terms in SEGMENT_TERMS.items() if any(t in body for t in terms)]


def main():
    docs = load_documents()
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i

    by_segment: dict[str, list] = {seg: [] for seg in SEGMENT_TERMS}
    for d in docs:
        text = d.page_content
        nums = numbers_in(text)
        if not nums or not any(w in text.lower() for w in MARKET_WORDS):
            continue
        for seg in segments_of(text) or ["기타"]:
            by_segment[seg].append((d, nums))

    lines = ["# 시장 수치 후보 청크 (자동 추출, 정답지 작성용)", "",
             f"전체 청크 {len(docs)}개. 금액·% + 시장 관련어가 함께 있는 청크만 모았다. "
             "한 청크가 여러 분야 용어를 담으면 여러 분야에 중복으로 나온다.", ""]
    for seg, items in by_segment.items():
        lines += [f"## {seg} ({len(items)}개)", ""]
        for d, nums in items:
            m = d.metadata
            lines += [f"### chunk {m['chunk_id']} · {m['doc_type']} · {m['source']} p.{m['page']}",
                      f"수치: {', '.join(dict.fromkeys(nums))}", "", "```", d.page_content.strip(), "```", ""]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"청크 {len(docs)}개 → 분야별 후보 " + ", ".join(f"{s} {len(v)}" for s, v in by_segment.items()))
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
