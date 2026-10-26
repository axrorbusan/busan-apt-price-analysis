"""
01. 실거래가 정제 (cleaning)

입력: data/raw/busan_apt_trades_raw.csv   (국토교통부 아파트 매매 실거래가 상세 자료, 부산 16개 구·군)
출력: data/processed/trades_clean.parquet  (분석용 거래)
      data/processed/complexes.csv         (단지 목록 — 지오코딩 입력)
      output/cleaning_log.md               (정제 단계별 건수)
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "busan_apt_trades_raw.csv")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")

SGG = {"26110": "중구", "26140": "서구", "26170": "동구", "26200": "영도구", "26230": "부산진구",
       "26260": "동래구", "26290": "남구", "26320": "북구", "26350": "해운대구", "26380": "사하구",
       "26410": "금정구", "26440": "강서구", "26470": "연제구", "26500": "수영구", "26530": "사상구",
       "26710": "기장군"}
PYEONG = 3.305785


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    raw = RAW if os.path.exists(RAW) else RAW + ".gz"       # 저장소에는 gzip 압축본 포함
    df = pd.read_csv(raw, dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())
    log = [("원본 거래 (raw)", len(df))]

    df = df.drop_duplicates()
    log.append(("완전 동일 레코드 제거 후", len(df)))

    # ---- 형 변환
    df["sgg_nm"] = df["sggCd"].map(SGG)
    df["price_man"] = pd.to_numeric(df["dealAmount"].str.replace(",", ""), errors="coerce")  # 만원
    df["area"] = pd.to_numeric(df["excluUseAr"], errors="coerce")                          # 전용면적 ㎡
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    df["build_year"] = pd.to_numeric(df["buildYear"], errors="coerce")
    df["deal_date"] = pd.to_datetime(df["dealYear"] + "-" + df["dealMonth"].str.zfill(2) + "-"
                                     + df["dealDay"].str.zfill(2), errors="coerce")
    df["ym"] = df["deal_date"].dt.to_period("M").astype(str)
    df["age"] = df["deal_date"].dt.year - df["build_year"]
    df["cancelled"] = df["cdealType"].ne("")
    df["direct"] = df["dealingGbn"].eq("직거래")
    df["leasehold"] = df["landLeaseholdGbn"].eq("Y")

    # ---- 제외 기준
    n = len(df)
    df = df[~df["cancelled"]]
    log.append(("계약 해제 거래 제외 (cdealType)", len(df)))
    df = df[~df["leasehold"]]
    log.append(("토지임대부 제외", len(df)))
    df = df.dropna(subset=["price_man", "area", "floor", "build_year", "deal_date"])
    df = df[(df["area"].between(15, 250)) & (df["floor"] >= 1) & (df["age"] >= -3)]
    log.append(("면적 15–250㎡, 지상층, 건축연도 정상", len(df)))

    df["ppm2"] = df["price_man"] * 10000 / df["area"]          # 원/㎡
    df["ppp"] = df["price_man"] / (df["area"] / PYEONG)          # 만원/평

    # ---- 단지·연도 내 이상치 (중위값 대비 ±50% log 편차 ≈ 0.6배~1.65배 밖)
    df["year"] = df["deal_date"].dt.year
    med = df.groupby(["aptSeq", "year"])["ppm2"].transform("median")
    cnt = df.groupby(["aptSeq", "year"])["ppm2"].transform("size")
    dev = np.log(df["ppm2"] / med)
    df["outlier"] = (cnt >= 3) & (dev.abs() > 0.5)
    log.append(("단지·연도 이상치 표시 (분석 제외)", int(df["outlier"].sum())))

    # 직거래 vs 중개거래: 같은 단지·연도 중위값 대비 비율
    ok3 = cnt >= 3
    low_direct = float((dev[df["direct"] & ok3] < np.log(0.8)).mean())
    low_broker = float((dev[~df["direct"] & ok3] < np.log(0.8)).mean())
    share_direct = float(df["direct"].mean())

    df = df.drop(columns=["dealAmount", "excluUseAr", "buildYear", "landLeaseholdGbn"])
    df.to_parquet(os.path.join(PROC, "trades_clean.parquet"), index=False)

    # ---- 단지 목록 (지오코딩 입력)
    cx = (df.groupby("aptSeq")
            .agg(apt_nm=("aptNm", "first"), sgg_nm=("sgg_nm", "first"), umd_nm=("umdNm", "first"),
                 jibun=("jibun", "first"), road_nm=("roadNm", "first"),
                 road_bon=("roadNmBonbun", "first"), road_bu=("roadNmBubun", "first"),
                 build_year=("build_year", "first"), n_trades=("ppm2", "size"))
            .reset_index())
    cx["addr_jibun"] = "부산 " + cx["sgg_nm"] + " " + cx["umd_nm"] + " " + cx["jibun"]
    def road(r):
        if not r["road_nm"] or not r["road_bon"]:
            return ""
        b = int(r["road_bon"]); s = int(r["road_bu"] or 0)
        return f"부산 {r['sgg_nm']} {r['road_nm']} {b}" + (f"-{s}" if s else "")
    cx["addr_road"] = cx.apply(road, axis=1)
    cx.to_csv(os.path.join(PROC, "complexes.csv"), index=False, encoding="utf-8-sig")

    lines = ["# 정제 로그 (Cleaning log)", "", "| 단계 | 건수 |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in log]
    lines += ["", f"- 분석 기간: {df['deal_date'].min():%Y-%m} ~ {df['deal_date'].max():%Y-%m}",
              f"- 단지 수: {len(cx):,}",
              f"- 직거래 비중: {share_direct:.1%}",
              f"- 같은 단지·연도 중위값보다 20% 이상 싸게 거래된 비율: 직거래 {low_direct:.1%} vs 중개거래 {low_broker:.1%}",
              "  → 직거래에는 시장가와 다른 거래(가족 간 거래 등)가 더 많이 섞여 있어 모형에서 제외",
              "- 참고: '완전 중복'은 모든 항목이 같은 레코드로, 같은 날 같은 층 같은 가격의 서로 다른 세대일 수도 있으나 동(棟) 정보가 비어 있어 구분할 수 없으므로 1건만 남김"]
    with open(os.path.join(OUT, "cleaning_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
