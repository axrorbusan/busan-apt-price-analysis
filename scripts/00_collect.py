"""
00. 데이터 수집 (data collection)

1) 국토교통부 아파트 매매 실거래가 상세 자료 — 부산 16개 구·군 × 월
2) 단지 좌표 — 카카오 로컬 API (지번 주소 → 도로명 주소 → 키워드 순으로 시도)
3) 지하철역·학교 위치 — 카카오 로컬 카테고리 검색 (SW8, SC4), 영역 분할 검색

키 파일: keys.txt (DATA_GO_KR=..., KAKAO_REST=...) — 저장소에 포함하지 않음
사용법:
  python scripts/00_collect.py trades      # → data/raw/busan_apt_trades_raw.csv
  python scripts/00_collect.py geocode     # → data/raw/kakao_geocodes.csv (01_clean.py 실행 후)
  python scripts/00_collect.py poi         # → data/raw/busan_subway_stations_kakao.csv, busan_schools_kakao.csv

참고: 포트폴리오 작성 시에는 같은 로직을 브라우저에서 실행해 수집했다
(작업 환경의 네트워크 제한 때문). API 초당 호출 제한이 있어 요청 간 간격을 둔다.
"""
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)

SGG = ["26110", "26140", "26170", "26200", "26230", "26260", "26290", "26320", "26350",
       "26380", "26410", "26440", "26470", "26500", "26530", "26710"]
COLS = ["sggCd", "umdCd", "umdNm", "aptNm", "aptSeq", "aptDong", "jibun", "bonbun", "bubun",
        "roadNm", "roadNmBonbun", "roadNmBubun", "buildYear", "excluUseAr", "floor", "dealYear",
        "dealMonth", "dealDay", "dealAmount", "dealingGbn", "buyerGbn", "slerGbn", "cdealType",
        "cdealDay", "rgstDate", "landLeaseholdGbn"]
TRADE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
KAKAO = "https://dapi.kakao.com/v2/local"


def keys():
    path = os.environ.get("KEYS_FILE", P("..", "keys.txt"))
    if not os.path.exists(path):
        path = P("keys.txt")
    return dict(l.strip().split("=", 1) for l in open(path, encoding="utf-8") if "=" in l)


def months(start=(2021, 1), end=(2026, 9)):
    y, m = start
    while (y, m) <= end:
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_trades():
    key = keys()["DATA_GO_KR"]
    out = P("data", "raw", "busan_apt_trades_raw.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for sgg in SGG:
            for ym in months():
                page = 1
                while True:
                    for attempt in range(8):
                        time.sleep(0.7)
                        r = requests.get(TRADE_URL, params=dict(serviceKey=key, LAWD_CD=sgg, DEAL_YMD=ym,
                                                                numOfRows=1000, pageNo=page), timeout=30)
                        if "PER_SECOND" in r.text:          # 초당 호출 제한 → 대기 후 재시도
                            time.sleep(15)
                            continue
                        root = ET.fromstring(r.content)
                        if root.findtext(".//resultCode") == "000":
                            break
                        time.sleep(5)
                    else:
                        print("FAILED", sgg, ym, page)
                        break
                    for it in root.iter("item"):
                        w.writerow([(it.findtext(c) or "").strip() for c in COLS])
                    total = int(root.findtext(".//totalCount") or 0)
                    if page * 1000 >= total:
                        break
                    page += 1
            print("done", sgg)


def kakao_get(path, **params):
    h = {"Authorization": f"KakaoAK {keys()['KAKAO_REST']}"}
    for _ in range(4):
        r = requests.get(f"{KAKAO}/{path}", headers=h, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        time.sleep(1)
    return {"documents": [], "meta": {"total_count": 0, "is_end": True}}


def geocode():
    cx = pd.read_csv(P("data", "processed", "complexes.csv"), dtype=str, keep_default_na=False)
    out = []
    for _, r in cx.iterrows():
        lon = lat = None
        method = "실패"
        for m, q, path in (("지번주소", r["addr_jibun"], "search/address.json"),
                           ("도로명주소", r["addr_road"], "search/address.json"),
                           ("키워드(단지명)", f"부산 {r['sgg_nm']} {r['umd_nm']} {r['apt_nm']} 아파트", "search/keyword.json")):
            if not q:
                continue
            d = kakao_get(path, query=q).get("documents", [])
            if d:
                lon, lat, method = float(d[0]["x"]), float(d[0]["y"]), m
                break
        out.append(dict(aptSeq=r["aptSeq"], lon=lon, lat=lat, geocode_method=method))
        time.sleep(0.05)
    pd.DataFrame(out).to_csv(P("data", "raw", "kakao_geocodes.csv"), index=False, encoding="utf-8-sig")


def poi():
    def rect(code, x1, y1, x2, y2, depth, acc):
        docs, page, total = [], 1, 0
        while page <= 3:
            j = kakao_get("search/category.json", category_group_code=code,
                          rect=f"{x1},{y1},{x2},{y2}", page=page, size=15)
            total = j["meta"]["total_count"]
            docs += j["documents"]
            if j["meta"]["is_end"]:
                break
            page += 1
        if total > 45 and depth < 7:                     # 45건 제한 → 영역 4분할
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            for a in ((x1, y1, mx, my), (mx, y1, x2, my), (x1, my, mx, y2), (mx, my, x2, y2)):
                rect(code, *a, depth + 1, acc)
            return
        for d in docs:
            acc[d["id"]] = dict(kakao_id=d["id"], name=d["place_name"], category=d["category_name"],
                                address=d["address_name"], road_address=d["road_address_name"],
                                lon=d["x"], lat=d["y"])

    for code, name in (("SW8", "busan_subway_stations_kakao.csv"), ("SC4", "busan_schools_kakao.csv")):
        acc, step, x = {}, 0.05, 128.75
        while x < 129.32:
            y = 34.95
            while y < 35.40:
                rect(code, x, y, x + step, y + step, 0, acc)
                y += step
            x += step
        pd.DataFrame(acc.values()).to_csv(P("data", "raw", name), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    {"trades": fetch_trades, "geocode": geocode, "poi": poi}[sys.argv[1]]()
