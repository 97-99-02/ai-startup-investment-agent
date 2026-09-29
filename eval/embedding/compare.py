"""임베딩 후보 비교 실험 (설계 문서 2.3)

1) rag.ingest.load_documents()로 실제 파이프라인과 같은 청크를 만든다
2) 문서 유형(tech/market)별로 청크를 무작위로 뽑아, LLM이 "그 청크로만 답할 수 있는 질문"을 만든다
   → (질문, 정답 청크 id). 평가 세트는 qa_set.json에 저장해 모든 모델을 같은 질문으로 비교한다
3) 모델마다 청크와 질문을 임베딩하고, 설계대로 같은 문서 유형 안에서만 검색해 정답 청크의 순위를 본다
4) Hit Rate@K, MRR@5, 문서 유형별 Hit Rate@5, 임베딩 시간을 results.json에 저장한다

실행 (프로젝트 루트에서): uv run python -m eval.embedding.compare
"""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from rag.embeddings import _device
from rag.ingest import load_documents

load_dotenv()
HERE = Path(__file__).resolve().parent
QA_PATH = HERE / "qa_set.json"
RESULT_PATH = HERE / "results.json"

CANDIDATES = ["BAAI/bge-m3", "nlpai-lab/KURE-v1", "Qwen/Qwen3-Embedding-0.6B"]
QA_MODEL = "gpt-4.1-mini"
N_QUESTIONS_PER_TYPE = 25
KS = (1, 3, 5)
SEED = 42


class QA(BaseModel):
    question: str


def build_qa_set(docs) -> list[dict]:
    if QA_PATH.exists():  # 한 번 만든 평가 세트를 재사용해 모델 간 비교 조건을 같게 한다
        return json.loads(QA_PATH.read_text(encoding="utf-8"))
    llm = ChatOpenAI(model=QA_MODEL, temperature=0).with_structured_output(QA)
    rng = random.Random(SEED)
    qa = []
    for doc_type in ("tech", "market"):
        pool = [d for d in docs if d.metadata["doc_type"] == doc_type and len(d.page_content) > 300]
        for d in rng.sample(pool, N_QUESTIONS_PER_TYPE):
            q = llm.invoke(
                "다음은 AI 반도체 보고서의 한 부분이다. 이 부분의 내용으로만 답할 수 있는 구체적인 질문을 "
                "한국어로 하나 만들어라. 투자 검토자가 실제로 물을 법한 질문이어야 하고, 본문 문장을 그대로 "
                f"옮기지 말 것.\n\n{d.page_content}"
            ).question
            qa.append({"question": q, "gold_id": d.metadata["chunk_id"], "doc_type": doc_type,
                       "source": d.metadata["source"], "page": d.metadata["page"]})
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return qa


def evaluate(model_name: str, docs, qa) -> dict:
    model = SentenceTransformer(model_name, device=_device())
    query_prompt = (model.prompts or {}).get("query", "")
    query_kwargs = {"prompt_name": "query"} if query_prompt else {}

    t0 = time.time()
    doc_emb = model.encode([d.page_content for d in docs], batch_size=8, normalize_embeddings=True)
    encode_sec = time.time() - t0
    q_emb = model.encode([x["question"] for x in qa], normalize_embeddings=True, **query_kwargs)

    types = np.array([d.metadata["doc_type"] for d in docs])
    ranks = []
    for qv, x in zip(q_emb, qa):
        sims = doc_emb @ qv
        sims[types != x["doc_type"]] = -np.inf  # 같은 문서 유형 안에서만 검색 (설계 2.4 필터)
        top = list(np.argsort(-sims)[: max(KS)])
        ranks.append(top.index(x["gold_id"]) + 1 if x["gold_id"] in top else None)

    def hit(k, subset=None):
        idx = [i for i, x in enumerate(qa) if subset is None or x["doc_type"] == subset]
        return round(sum(ranks[i] is not None and ranks[i] <= k for i in idx) / len(idx), 3)

    result = {
        "model": model_name,
        "query_prompt": query_prompt,
        **{f"hit@{k}": hit(k) for k in KS},
        "mrr@5": round(sum(1 / r for r in ranks if r) / len(qa), 3),
        "hit@5_tech": hit(5, "tech"),
        "hit@5_market": hit(5, "market"),
        "encode_sec": round(encode_sec, 1),
        "dim": int(doc_emb.shape[1]),
    }
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def main():
    docs = load_documents()
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i
    qa = build_qa_set(docs)
    n_tech = sum(d.metadata["doc_type"] == "tech" for d in docs)
    print(f"청크 {len(docs)}개 (tech {n_tech}, market {len(docs) - n_tech}), 평가 질문 {len(qa)}개, device={_device()}")
    results = []
    for name in CANDIDATES:
        r = evaluate(name, docs, qa)
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
    RESULT_PATH.write_text(json.dumps({
        "setup": {"chunks": len(docs), "questions": len(qa), "qa_model": QA_MODEL,
                  "chunk_size": 800, "chunk_overlap": 100, "device": _device()},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
