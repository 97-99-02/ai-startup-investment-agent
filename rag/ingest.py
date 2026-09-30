"""RAG 문서 적재 (설계 2.2, 2.4): data/tech, data/market PDF → 청크 → 임베딩 → Chroma

실행: uv run python -m rag.ingest   (문서를 바꾼 뒤 한 번만 실행. 결과는 vectorstore/에 저장)
"""
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHROMA_DIR, DOC_TYPES
from rag.catalog import lookup
from rag.ocr_fix import ocr_path
from rag.embeddings import get_embeddings

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "ai_semiconductor"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100

# 숫자·기호만 있는 줄 (예: "1,800", "524", "'23(E)", "-15%")
_NUMBER_ONLY = re.compile(r"""^[\s\d.,%()\[\]'"‘’“”+\-~/EF]*$""")
CHART_RUN = 3


def drop_chart_numbers(text: str) -> str:
    """OCR 텍스트에서 그래프 눈금·값을 뺀다.

    OCR은 그래프 숫자를 위치 순서대로 한 줄씩 읽어, 어느 연도·분야의 값인지 알 수 없게 뒤섞인다
    (KDB 4쪽 그래프에서 시장성 평가가 "PC용 155억→524억 달러"를 잘못 뽑아 사실 검증에 걸렸다).
    숫자만 있는 줄이 CHART_RUN줄 이상 이어지면 그래프로 보고 뺀다. 행 단위로 읽힌 표는 숫자와 글자 줄이 번갈아 나와
    남지만, 열 단위로 읽힌 표는 함께 빠진다(KDB 21쪽 표 9).
    """
    kept, run = [], []
    for line in text.splitlines() + [None]:
        if line is not None and line.strip() and _NUMBER_ONLY.match(line):
            run.append(line)
            continue
        if len(run) < CHART_RUN:
            kept.extend(run)
        run = []
        if line is not None:
            kept.append(line)
    return "\n".join(kept)


def load_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for doc_type in DOC_TYPES:
        for pdf in sorted((ROOT / "data" / doc_type).glob("*.pdf")):
            info = lookup(pdf.name)  # 제목·발행기관·연도·URL (설계 2.4 메타데이터)
            for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
                fixed = ocr_path(pdf.name, page_no)  # 글꼴 손상 쪽은 OCR 텍스트를 쓴다 (rag/ocr_fix.py)
                if fixed.exists():
                    text = drop_chart_numbers(fixed.read_text(encoding="utf-8"))
                else:
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
