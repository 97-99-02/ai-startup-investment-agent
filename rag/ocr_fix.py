"""글꼴 손상으로 텍스트가 깨지는 PDF 쪽을 OCR로 다시 읽어 data/ocr/ 에 저장한다 (한 번만 실행하는 전처리).

KDB 보고서는 Type 3 글꼴 문제로 pypdf·pdftotext·PyMuPDF 모두 일부 숫자와 영문을 깨뜨린다.
예: 원문 "'26년 3,400백만달러 ... YOLE('23.9)" → 추출 "'2/HxT67년 ... (/HxT'p/HxT55LE"
이런 쪽만 이미지로 바꿔 macOS 기본 OCR(Apple Vision)로 읽고, rag/ingest.py가 그 쪽은 OCR 텍스트를 쓴다.
결과 텍스트를 저장소에 함께 올리므로 다른 환경에서는 이 스크립트 없이 ingest만 실행하면 된다.

실행 (macOS, 프로젝트 루트): uv run --with ocrmac --with pymupdf python -m rag.ocr_fix
"""
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "data" / "ocr"
DAMAGED = re.compile(r"/H[A-Za-z]{1,2}T")  # 손상된 글꼴 글리프가 텍스트에 섞여 나오는 표시 (예: /HxT, /HFT)


def ocr_path(pdf_name: str, page_no: int) -> Path:
    return OCR_DIR / Path(pdf_name).stem / f"p{page_no:03d}.txt"


def find_damaged_pages() -> dict[Path, list[int]]:
    found = {}
    for pdf in sorted((ROOT / "data").glob("*/*.pdf")):
        pages = [i for i, p in enumerate(PdfReader(pdf).pages, start=1) if DAMAGED.search(p.extract_text() or "")]
        if pages:
            found[pdf] = pages
    return found


def main():
    import pymupdf
    from ocrmac import ocrmac

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OCR_DIR / "_page.png"
    for pdf, pages in find_damaged_pages().items():
        doc = pymupdf.open(pdf)
        for page_no in pages:
            doc[page_no - 1].get_pixmap(dpi=200).save(tmp)
            lines = ocrmac.OCR(str(tmp), language_preference=["ko-KR", "en-US"], recognition_level="accurate").recognize()
            out = ocr_path(pdf.name, page_no)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(line[0] for line in lines), encoding="utf-8")
        print(f"{pdf.name}: {len(pages)}쪽 OCR {pages}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
