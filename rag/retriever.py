"""에이전트용 검색기 (설계 2.4): 문서 유형(tech/market) 메타데이터로 필터링해 검색한다.

사용 예
    from rag.retriever import get_retriever, to_source
    docs = get_retriever("tech").invoke("HBM 메모리 대역폭이 NPU 성능에 주는 영향")
    for d in docs: d.page_content, d.metadata["title"], d.metadata["publisher"], d.metadata["page"]
    sources = [to_source(d, company="모빌린트", node="tech_summary") for d in docs]  # state.Source 형식
"""
import threading
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, DOC_TYPES
from rag.embeddings import get_embeddings
from rag.ingest import COLLECTION

ROOT = Path(__file__).resolve().parent.parent
# 병렬 노드가 처음에 동시에 부르면 Chroma 클라이언트를 두 번 만들다 KeyError가 난다. 처음 한 번만 만들도록 잠근다
_LOCK = threading.Lock()


def get_vectorstore() -> Chroma:
    with _LOCK:
        return _load_vectorstore()


@lru_cache(maxsize=1)
def _load_vectorstore() -> Chroma:
    persist = ROOT / CHROMA_DIR
    if not persist.exists():
        raise FileNotFoundError("vectorstore가 없습니다. 먼저 `uv run python -m rag.ingest` 를 실행하세요.")
    return Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(), persist_directory=str(persist))


def get_retriever(doc_type: str, k: int = 4):
    assert doc_type in DOC_TYPES, f"doc_type은 {DOC_TYPES} 중 하나"
    return get_vectorstore().as_retriever(search_kwargs={"k": k, "filter": {"doc_type": doc_type}})


def document_source_id(doc: Document) -> str:
    """문서명·페이지·본문 해시로 근거를 식별한다."""
    digest = sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"report:{doc.metadata.get('source', '')}:p{doc.metadata.get('page', '')}:{digest}"


def to_source(doc, company: str, node: str) -> dict:
    """검색된 청크를 state.Source 형식(kind="report")으로 바꾼다. 보고서 REFERENCE에 그대로 쓰인다."""
    m = doc.metadata
    source = {
        "company": company,
        "node": node,
        "kind": "report",
        "source_id": document_source_id(doc),
        "source_file": m.get("source", ""),
        "source_path": m.get("source_path") or (
            f"data/{m['doc_type']}/{m['source']}"
            if m.get("doc_type") and m.get("source") else ""
        ),
        "title": m.get("title", m.get("source", "")),
        "publisher": m.get("publisher", ""),
        "date": m.get("year", ""),
        "url": m.get("url", ""),
        "snippet": doc.page_content[:800],
        "page": m.get("page"),
    }
    return source
