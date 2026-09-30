"""웹 검색 공통 도우미 (Tavily). 검색 결과를 state.Source 형식 출처로 바꾸는 함수도 둔다."""
import re
from datetime import date
from email.utils import parsedate_to_datetime

from langchain_tavily import TavilySearch


def search(query: str, max_results: int = 8, topic: str = "news") -> list[dict]:
    """Tavily 검색 결과 리스트. 각 항목: title, url, content, published_date"""
    response = TavilySearch(max_results=max_results, topic=topic).invoke({"query": query})
    # 사용량 초과(Error 432) 등은 예외가 아니라 {"error": ...}로 돌아온다. 빈 결과로 넘기면 모든 에이전트가
    # 근거 없이 계속 진행하므로 여기서 멈춘다
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Tavily 검색 실패: {response['error']}")
    return response.get("results", [])


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


# 기사 URL에 들어 있는 날짜: /2024/02/18/ 형식, 또는 20240104083305처럼 날짜로 시작하는 10자리 이상 기사 번호
_URL_DATE_PATH = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})(?:/|$)")
_URL_DATE_ID = re.compile(r"[=/](20\d{2})(\d{2})(\d{2})\d{2,}(?!\d)")


def date_from_url(url: str) -> str:
    """URL에서 게시일(YYYY-MM-DD)을 찾는다. 달력에 없는 날짜나 미래 날짜는 기사 번호로 보고 버린다."""
    for pattern in (_URL_DATE_PATH, _URL_DATE_ID):
        for y, m, d in pattern.findall(url):
            try:
                found = date(int(y), int(m), int(d))
            except ValueError:
                continue
            if found <= date.today():
                return found.isoformat()
    return ""


def source_date(r: dict) -> str:
    """검색 결과의 게시일. Tavily 게시일(뉴스 검색)이 없으면 URL에서 찾고, 둘 다 없으면 빈 문자열."""
    try:
        return parsedate_to_datetime(r["published_date"]).strftime("%Y-%m-%d")
    except (KeyError, TypeError, ValueError):
        return date_from_url(r.get("url", ""))  # 일반 웹 검색은 게시일을 주지 않는다


def to_source(r: dict, company: str, node: str) -> dict:
    """Tavily 결과 하나를 Source 형식으로. 언론사는 제목 끝 ' - 언론사' 에서 추출한다."""
    title, _, publisher = r["title"].rpartition(" - ")
    return {
        "company": company,
        "node": node,
        "kind": "web",
        "title": (title or r["title"]).strip(),
        "publisher": publisher.strip() if title else "",
        "date": source_date(r),
        "url": r["url"],
        "snippet": r["content"][:800],
    }
