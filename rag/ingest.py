"""RAG 문서 적재 (설계 2.2, 2.4): data/tech, data/market PDF → 청크 → 임베딩 → Chroma

실행: uv run python -m rag.ingest   (문서를 바꾼 뒤 한 번만 실행. 결과는 vectorstore/에 저장)
"""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHROMA_DIR, DOC_TYPES
from rag.catalog import lookup
from rag.embeddings import get_embeddings

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "ai_semiconductor"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100


def load_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for doc_type in DOC_TYPES:
        for pdf in sorted((ROOT / "data" / doc_type).glob("*.pdf")):
            info = lookup(pdf.name)  # 제목·발행기관·연도·URL (설계 2.4 메타데이터)
            for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
                text = page.extract_text() or ""
                for chunk in splitter.split_text(text):
                    if len(chunk.strip()) < 100:  # 쪽번호·머리말 조각 제외
                        continue
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"doc_type": doc_type, "source": pdf.name,
                                  "source_path": str(pdf.relative_to(ROOT)), "page": page_no,
                                  "title": info["title"], "publisher": info["publisher"],
                                  "year": info["year"], "url": info["url"]},
                    ))
    return docs


def main():
    persist = ROOT / CHROMA_DIR
    if persist.exists():
        shutil.rmtree(persist)
    docs = load_documents()
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i  # 검색 평가(Hit Rate, MRR)용 id
    Chroma.from_documents(docs, get_embeddings(), collection_name=COLLECTION, persist_directory=str(persist))
    counts = {t: sum(d.metadata["doc_type"] == t for d in docs) for t in DOC_TYPES}
    print(f"청크 {len(docs)}개 저장 {counts} → {persist}")


if __name__ == "__main__":
    main()
