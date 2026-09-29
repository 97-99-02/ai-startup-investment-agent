"""웹 검색 공통 도우미 (Tavily). 검색 결과를 state.Source 형식 출처로 바꾸는 함수도 둔다."""
from email.utils import parsedate_to_datetime

from langchain_tavily import TavilySearch


def search(query: str, max_results: int = 8, topic: str = "news") -> list[dict]:
    """Tavily 검색 결과 리스트. 각 항목: title, url, content, published_date"""
    return TavilySearch(max_results=max_results, topic=topic).invoke({"query": query}).get("results", [])


def search_many(queries: list[str], max_results: int = 8, topic: str = "news") -> list[dict]:
    """여러 검색어 결과를 url 기준으로 중복 제거해 합친다."""
    seen, merged = set(), []
    for q in queries:
        for r in search(q, max_results, topic):
            if r["url"] not in seen:
                seen.add(r["url"])
                merged.append(r)
    return merged


def format_results(results: list[dict], limit: int = 1200) -> str:
    """LLM 프롬프트에 넣을 형태. [번호]로 결과를 구분해 근거 url을 고를 수 있게 한다."""
    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content'][:limit]}" for i, r in enumerate(results)
    )


def _date(r: dict) -> str:
    try:
        return parsedate_to_datetime(r["published_date"]).strftime("%Y-%m-%d")
    except (KeyError, TypeError, ValueError):
        return ""  # 일반 웹 검색은 게시일이 없는 경우가 있다


def to_source(r: dict, company: str, node: str) -> dict:
    """Tavily 결과 하나를 Source 형식으로. 언론사는 제목 끝 ' - 언론사' 에서 추출한다."""
    title, _, publisher = r["title"].rpartition(" - ")
    return {
        "company": company,
        "node": node,
        "kind": "web",
        "title": (title or r["title"]).strip(),
        "publisher": publisher.strip() if title else "",
        "date": _date(r),
        "url": r["url"],
        "snippet": r["content"][:800],
    }
