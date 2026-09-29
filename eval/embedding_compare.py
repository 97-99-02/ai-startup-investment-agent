"""설계 문서 2.3 임베딩 후보 비교 실험 (Hit Rate@K, MRR@K)

1) data/tech, data/market PDF를 청크로 나눈다
2) 문서 유형별로 청크를 뽑아 LLM이 "그 청크로만 답할 수 있는 질문"을 만든다 → (질문, 정답 청크 id)
   평가 세트는 eval/qa_set.json 에 저장해 세 모델을 같은 질문으로 비교한다
3) 모델마다 청크와 질문을 임베딩하고, 설계대로 같은 문서 유형 안에서만 검색해 순위를 매긴다

실행: python eval/embedding_compare.py  (프로젝트 루트에서)
"""
import json
import random
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "eval" / "qa_set.json"
RESULT_PATH = ROOT / "eval" / "embedding_result.json"

CANDIDATES = {
    "BAAI/bge-m3": {},
    "nlpai-lab/KURE-v1": {},
    "Qwen/Qwen3-Embedding-0.6B": {"query_prompt_name": "query"},
}
N_QUESTIONS_PER_TYPE = 25
KS = (1, 3, 5)
SEED = 42


def load_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for doc_type in ("tech", "market"):
        for pdf in sorted((ROOT / "data" / doc_type).glob("*.pdf")):
            text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
            for piece in splitter.split_text(text):
                if len(piece.strip()) > 200:  # 표지·목차 조각 제외
                    chunks.append({"id": len(chunks), "doc_type": doc_type, "source": pdf.name, "text": piece})
    return chunks


class QA(BaseModel):
    question: str


def build_qa_set(chunks):
    if QA_PATH.exists():
        return json.loads(QA_PATH.read_text(encoding="utf-8"))
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(QA)
    rng = random.Random(SEED)
    qa = []
    for doc_type in ("tech", "market"):
        pool = [c for c in chunks if c["doc_type"] == doc_type]
        for c in rng.sample(pool, N_QUESTIONS_PER_TYPE):
            q = llm.invoke(
                "다음은 AI 반도체 보고서의 한 부분이다. 이 부분의 내용으로만 답할 수 있는 구체적인 질문을 "
                "한국어로 하나 만들어라. 투자 검토자가 실제로 물을 법한 질문이어야 하고, 본문 문장을 그대로 "
                f"베끼지 말 것.\n\n{c['text']}"
            ).question
            qa.append({"question": q, "gold_id": c["id"], "doc_type": doc_type})
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return qa


def evaluate(model_name, opts, chunks, qa):
    t0 = time.time()
    model = SentenceTransformer(model_name, device="mps")
    load_s = time.time() - t0
    t0 = time.time()
    doc_emb = model.encode([c["text"] for c in chunks], batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False)
    encode_s = time.time() - t0
    q_kwargs = {"prompt_name": opts["query_prompt_name"]} if "query_prompt_name" in opts else {}
    q_emb = model.encode([x["question"] for x in qa], normalize_embeddings=True, **q_kwargs)

    types = np.array([c["doc_type"] for c in chunks])
    hits = {k: 0 for k in KS}
    rr = 0.0
    per_type = {t: {"hit5": 0, "n": 0} for t in ("tech", "market")}
    for qv, x in zip(q_emb, qa):
        sims = doc_emb @ qv
        sims[types != x["doc_type"]] = -1  # 같은 문서 유형(컬렉션) 안에서만 검색
        ranking = list(np.argsort(-sims)[: max(KS)])
        per_type[x["doc_type"]]["n"] += 1
        if x["gold_id"] in ranking:
            rank = ranking.index(x["gold_id"]) + 1
            rr += 1 / rank
            for k in KS:
                hits[k] += rank <= k
            per_type[x["doc_type"]]["hit5"] += 1
    n = len(qa)
    return {
        "model": model_name,
        **{f"hit@{k}": round(hits[k] / n, 3) for k in KS},
        f"mrr@{max(KS)}": round(rr / n, 3),
        "hit@5_tech": round(per_type["tech"]["hit5"] / per_type["tech"]["n"], 3),
        "hit@5_market": round(per_type["market"]["hit5"] / per_type["market"]["n"], 3),
        "load_sec": round(load_s, 1),
        "encode_sec": round(encode_s, 1),
    }


def main():
    chunks = load_chunks()
    qa = build_qa_set(chunks)
    print(f"청크 {len(chunks)}개 (tech {sum(c['doc_type']=='tech' for c in chunks)}, "
          f"market {sum(c['doc_type']=='market' for c in chunks)}), 평가 질문 {len(qa)}개")
    results = []
    for name, opts in CANDIDATES.items():
        r = evaluate(name, opts, chunks, qa)
        print(r)
        results.append(r)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
