"""웹 검색 공통 도우미 테스트 (API 키·네트워크 없이 실행): uv run --with pytest python -m pytest tests/test_web.py -q"""
from agents.web import date_from_url, date_in_page


def test_page_date_from_meta_tags():
    # 실제 언론사 페이지에서 날짜가 들어 있던 모양 (매일경제·뉴스1 / 매일경제TV / 다음 뉴스)
    assert date_in_page('<meta property="article:published_time" content="2026-06-03T10:00:00+09:00">') == "2026-06-03"
    assert date_in_page("<meta property='article:published'  content='2026-06-03 09:12'>") == "2026-06-03"
    assert date_in_page('{"title":"기사", regdate:"2025-05-15T09:00:00"}') == "2025-05-15"


def test_page_date_from_body_text_only_without_meta():
    assert date_in_page("<span>입력시간 | 2026.07.04 09:00</span>") == "2026-07-04"
    assert date_in_page("<span>입력 : 2026.06.03 11:20</span>") == "2026-06-03"
    # 메타 태그가 있으면 본문 날짜보다 먼저 쓴다
    html = '<span>입력 2025.01.01</span><meta property="article:published_time" content="2026-06-03">'
    assert date_in_page(html) == "2026-06-03"


def test_page_date_rejects_future_and_invalid():
    assert date_in_page('<meta property="article:published_time" content="2999-01-01">') == ""
    assert date_in_page("<span>입력 2026.13.40</span>") == ""
    assert date_in_page("<p>2026년 6월 3일 투자 유치</p>") == ""   # '입력'·메타 태그가 없는 본문 날짜는 쓰지 않는다


def test_url_date():
    assert date_from_url("https://wowtale.net/2026/05/30/259381") == "2026-05-30"
    assert date_from_url("https://www.etnews.com/20260601000245") == "2026-06-01"
    assert date_from_url("https://www.mk.co.kr/news/business/12064949") == ""
