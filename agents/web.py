"""웹 검색 공통 도우미 (Tavily). 검색 결과를 state.Source 형식 출처로 바꾸는 함수도 둔다."""
import re
import urllib.request
from datetime import date, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from langchain_tavily import TavilySearch


def search(query: str, max_results: int = 8, topic: str = "news") -> list[dict]:
    """Tavily 검색 결과 리스트. 각 항목: title, url, content, published_date"""
    response = TavilySearch(max_results=max_results, topic=topic).invoke({"query": query})
    # 결과가 0건이면 langchain-tavily가 ToolException을 문자열로 돌려준다(handle_tool_error=True).
    # 작은 스타트업의 좁은 검색어에서 자주 생기며, 오류가 아니라 '결과 없음'으로 처리한다
    if isinstance(response, str):
        return []
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
    """LLM 프롬프트에 넣을 형태. [번호]로 결과를 구분해 근거 url을 고를 수 있게 한다.
    게시일을 함께 넣는다. 날짜가 없으면 LLM이 "가장 최근 기사를 따른다"는 규칙을 지킬 수 없어,
    2026년 시리즈 B 기사 본문에 나온 '2024년 5월 시리즈 A' 날짜를 시리즈 B 시기로 뽑은 적이 있다."""
    return "\n\n".join(
        f"[{i}] {r['title']} ({source_date(r) or '날짜 미상'})\nURL: {r['url']}\n{r['content'][:limit]}"
        for i, r in enumerate(results)
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


# 기사 페이지의 게시일 메타 태그 (article:published_time, JSON-LD datePublished, 다음 뉴스의 regdate 등).
# 매일경제TV는 article:published로 끝나는 이름을 쓴다
_META_DATE = re.compile(r"""(?:article:published(?:_time)?|datePublished|pubdate|publish-date|dateCreated|regdate)["']?\s*"""
                        r"""(?:content|:)\s*=?\s*["'](20\d{2})[-./]?(\d{2})[-./]?(\d{2})""", re.I)
# 메타 태그가 없는 언론사는 본문 머리에 '입력 : 2026.06.03', '입력시간 | 2026.07.04'처럼 적는다 (이데일리TV)
_TEXT_DATE = re.compile(r"입력(?:시간|일)?\s*[:|]?\s*(20\d{2})[-.](\d{1,2})[-.](\d{1,2})")
# 매일경제는 'Mozilla/5.0'만 보내면 403으로 막는다
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


@lru_cache(maxsize=512)
def date_from_page(url: str) -> str:
    """기사 페이지를 읽어 게시일 메타 태그(없으면 본문의 '입력' 날짜)에서 날짜를 찾는다. 못 읽거나 없으면 빈 문자열.
    Tavily 게시일도 URL 날짜도 없어 보고서 REFERENCE에 '날짜 미상'이 많이 남았다. 메타 태그를 확인한
    기사는 Tavily 게시일과 모두 일치해 같은 기준으로 쓴다. 문서용 예약 도메인(example.org 등)은 요청하지 않는다."""
    host = urlparse(url).hostname or ""
    if not url.startswith(("http://", "https://")) or host.split(".")[-2:-1] == ["example"]:
        return ""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(request, timeout=4) as response:
            html = response.read(300_000).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001 - 날짜를 못 읽어도 출처 기록은 계속한다
        return ""
    return date_in_page(html)


def date_in_page(html: str) -> str:
    """페이지 HTML에서 게시일을 찾는다. 메타 태그를 먼저 보고, 없을 때만 본문의 '입력' 날짜를 쓴다."""
    for pattern in (_META_DATE, _TEXT_DATE):
        for y, m, d in pattern.findall(html):
            try:
                found = date(int(y), int(m), int(d))
            except ValueError:
                continue
            if found <= date.today():
                return found.isoformat()
    return ""


def source_date(r: dict, read_page: bool = False) -> str:
    """검색 결과의 게시일. Tavily 게시일(뉴스 검색) → URL 속 날짜 → (read_page=True면) 기사 페이지 메타 태그 순.
    페이지 읽기는 시간이 들어, 출처로 기록할 때(to_source)만 한다. 모두 없으면 빈 문자열."""
    try:
        published = parsedate_to_datetime(r["published_date"])
        if published.tzinfo is None:  # '-0000' 표기는 시간대 없는 UTC로 돌아온다
            published = published.replace(tzinfo=timezone.utc)
        day = published.astimezone(ZoneInfo("Asia/Seoul")).date()  # UTC로 자르면 한국 오전 기사가 전날로 찍힌다
        if day <= date.today():  # 미래 날짜는 게시일로 쓰지 않는다
            return day.isoformat()
    except (KeyError, TypeError, ValueError):
        pass  # 일반 웹 검색은 게시일을 주지 않는다
    return date_from_url(r.get("url", "")) or (date_from_page(r.get("url", "")) if read_page else "")


def to_source(r: dict, company: str, node: str) -> dict:
    """Tavily 결과 하나를 Source 형식으로. 언론사는 제목 끝 ' - 언론사' 에서 추출한다."""
    title, _, publisher = r["title"].rpartition(" - ")
    return {
        "company": company,
        "node": node,
        "kind": "web",
        "title": (title or r["title"]).strip(),
        "publisher": publisher.strip() if title else "",
        "date": source_date(r, read_page=True),
        "url": r["url"],
        "snippet": r["content"][:800],
    }
