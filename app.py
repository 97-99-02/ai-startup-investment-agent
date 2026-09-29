"""실행: uv run python app.py"""
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agents.reporter import export_report_pdf
from graph import build_graph

load_dotenv()


def main():
    app = build_graph()
    state = app.invoke({"current_index": 0, "retry_count": 0}, {"recursion_limit": 60})
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"report_{datetime.now():%Y%m%d_%H%M}.md"
    path.write_text(state["report"], encoding="utf-8")
    print(f"보고서 저장: {path}")
    pdf_path, pages = export_report_pdf(state["report"], state=state)
    print(f"제출용 PDF 저장: {pdf_path} ({pages}쪽)")
    print(f"평가 후보: {state['candidates']}")
    print(f"보류/제외: {[r['company'] for r in state.get('rejected', [])]}")


if __name__ == "__main__":
    main()
