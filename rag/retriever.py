"""에이전트용 검색기 (설계 2.4): 문서 유형(tech/market) 메타데이터로 필터링해 검색한다.

사용 예
    from rag.retriever import get_retriever
    docs = get_retriever("tech").invoke("HBM 메모리 대역폭이 NPU 성능에 주는 영향")
    for d in docs: d.page_content, d.metadata["source"], d.metadata["page"]
"""
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from config import CHROMA_DIR, DOC_TYPES
from rag.embeddings import get_embeddings
from rag.ingest import COLLECTION

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    persist = ROOT / CHROMA_DIR
    if not persist.exists():
        raise FileNotFoundError("vectorstore가 없습니다. 먼저 `uv run python -m rag.ingest` 를 실행하세요.")
    return Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(), persist_directory=str(persist))


def get_retriever(doc_type: str, k: int = 4):
    assert doc_type in DOC_TYPES, f"doc_type은 {DOC_TYPES} 중 하나"
    return get_vectorstore().as_retriever(search_kwargs={"k": k, "filter": {"doc_type": doc_type}})
