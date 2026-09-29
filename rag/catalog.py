"""RAG 문서 목록 (설계 문서 2.2). 제목은 각 PDF 첫 쪽의 원문 제목을 따른다.

rag/ingest.py가 파일명으로 이 목록을 찾아 청크 메타데이터(title, publisher, year, url)에 넣는다.
보고서 REFERENCE의 기관 보고서 형식 "발행기관(YYYY). 보고서명. URL" 에 그대로 쓰인다.
"""

ETRI = "한국전자통신연구원"
ETRI_URL = "https://ettrends.etri.re.kr/ettrends/"

DOCS = {
    # ---------- tech ----------
    "etri210_aichip_compiler.pdf": {
        "title": "최신 인공지능 반도체 및 컴파일러 지원 기술 동향",
        "publisher": ETRI, "year": "2024",
        "url": ETRI_URL + "210/0905210013/001-011.%20%EA%B9%80%EC%9A%A9%EC%A3%BC_210%ED%98%B8%20%EC%B5%9C%EC%A2%85.pdf",
    },
    "etri215_auto_aichip.pdf": {
        "title": "AI 정의 차량(ADV) 기반의 자율주행차용 AI반도체: 기술·시장 전망 및 전략적 시사점",
        "publisher": ETRI, "year": "2025",
        "url": ETRI_URL + "215/0905215002/012-024.%20%EC%86%A1%EA%B7%BC%ED%98%9C_215%ED%98%B8_%EC%B5%9C%EC%A2%85.pdf",
    },
    "etri216_chiplet_llm.pdf": {
        "title": "칩렛 이종집적 첨단패키지 기반 LLM 가속기 설계 동향",
        "publisher": ETRI, "year": "2025",
        "url": ETRI_URL + "216/0905216002/013-021.%20%EC%9E%A5%EB%AA%85%EC%9E%AC_216%ED%98%B8_%EC%B5%9C%EC%A2%85.pdf",
    },
    "etri217_physicalai_chip.pdf": {
        "title": "피지컬 AI 시대의 지각 특화 AI 반도체 개념 및 전략적 투자 방향",
        "publisher": ETRI, "year": "2025",
        "url": ETRI_URL + "217/0905217022/075-087.%20%EC%9D%B4%EC%84%B1%EC%A4%80_217%ED%98%B8_%EC%B5%9C%EC%A2%85.pdf",
    },
    "etri218_hpc_proc.pdf": {
        "title": "AI 워크로드를 위한 고성능 컴퓨팅(HPC) 프로세서 기술 발전 동향",
        "publisher": ETRI, "year": "2026",
        "url": ETRI_URL + "218/0905218007/069-081.%20%EB%B0%95%EC%9C%A0%EB%AF%B8_218%ED%98%B8_%EC%B5%9C%EC%A2%85.pdf",
    },
    "etri220_aimemory_market.pdf": {
        "title": "AI 메모리 반도체 분야 주요 기술 및 시장 동향",
        "publisher": ETRI, "year": "2026",
        "url": ETRI_URL + "220/0905220002/015-027.%20%ED%99%8D%EC%95%84%EB%A6%84_220%ED%98%B8_%EC%B5%9C%EC%A2%85.pdf",
    },
    "kisdi_outlook21_ondevice.pdf": {
        "title": "Physical AI 시대에 대응한 On-device AI 반도체 경쟁력 강화 방향",
        "publisher": "정보통신정책연구원", "year": "2025",
        "url": "https://www.kisdi.re.kr/report/fileDown.do?key=m2101113025377&arrMasterId=4333446&id=1839476",
    },
    # ---------- market ----------
    "exim_dc_aisemi_2026.pdf": {
        "title": "데이터센터용 AI반도체 시장 현황 및 전망",
        "publisher": "한국수출입은행 해외경제연구소", "year": "2026",
        "url": "https://keri.koreaexim.go.kr/comm/getFile?srvcId=BBSTY1&upperNo=115906&fileTy=ATTACH&fileNo=1",
    },
    "kdb_aisemi_tech_industry.pdf": {
        "title": "AI 반도체 기술 및 산업 동향",
        "publisher": "KDB미래전략연구소", "year": "2024",
        "url": "https://file.kdb.co.kr/fileView?groupId=E49EC0E6-F243-5B37-9586-D097FD724BD2&fileId=59D62619-83C0-01EC-F6FC-83BCAAC00856",
    },
    "kita_semi_2026.pdf": {
        "title": "반도체 전방산업 업황 진단 및 2026년 전망",
        "publisher": "한국무역협회", "year": "2025",
        "url": "https://www.kita.net/researchTrade/report/tradeBrief/tradeBriefDetail.do?no=2902",
    },
}


def lookup(file_name: str) -> dict:
    """파일명으로 문서 정보를 찾는다. 목록에 없는 파일이면 파일명을 제목으로 쓴다."""
    return DOCS.get(file_name, {"title": file_name, "publisher": "", "year": "", "url": ""})
