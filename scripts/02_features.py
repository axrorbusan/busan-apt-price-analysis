"""
02. 입지 변수 생성 (location features)

입력: data/processed/complexes.csv + data/raw/kakao_geocodes.csv (단지 좌표, 카카오 로컬 API)
      data/raw/busan_subway_stations_kakao.csv (카카오 '지하철역' 카테고리)
      data/raw/busan_schools_kakao.csv         (카카오 '학교' 카테고리)
      data/geo/sea.gpkg, data/geo/busan_districts.gpkg
출력: data/processed/trades_features.parquet
      output/geocoding_qa.md
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
CRS = 5179  # UTM-K (m)

CBD = {"seomyeon": (129.0597, 35.1579), "centum": (129.1310, 35.1690)}


def to_xy(lon, lat):
    g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(CRS)
    return np.c_[g.x, g.y]


def main():
    cx = pd.read_csv(P("data", "processed", "complexes.csv"), dtype={"aptSeq": str})
    geo = pd.read_csv(P("data", "raw", "kakao_geocodes.csv"), dtype={"aptSeq": str})
    cx = cx.merge(geo, on="aptSeq", how="left")
    cx["geocode_method"] = cx["geocode_method"].fillna("미시도")
    st = pd.read_csv(P("data", "raw", "busan_subway_stations_kakao.csv"))
    sc = pd.read_csv(P("data", "raw", "busan_schools_kakao.csv"))
    sea = gpd.read_file(P("data", "geo", "sea.gpkg")).to_crs(CRS).union_all()
    dist = gpd.read_file(P("data", "geo", "busan_districts.gpkg")).to_crs(CRS)

    # ---- 지오코딩 품질검사
    n = len(cx)
    ok = cx.dropna(subset=["lon", "lat"]).copy()
    xy = to_xy(ok["lon"], ok["lat"])
    ok["x"], ok["y"] = xy[:, 0], xy[:, 1]
    pts = gpd.GeoDataFrame(ok, geometry=gpd.points_from_xy(ok.x, ok.y), crs=CRS)
    j = gpd.sjoin(pts, dist[["sggnm", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    ok["sgg_check"] = j["sggnm"]
    mismatch = ok["sgg_check"].ne(ok["sgg_nm"])
    qa = ["# 지오코딩 품질검사 (Geocoding QA)", "",
          f"- 단지 수: {n:,}",
          f"- 좌표 획득: {len(ok):,} ({len(ok)/n:.2%})"]
    for m, c in ok["geocode_method"].value_counts().items():
        qa.append(f"  - {m}: {c:,}")
    qa += [f"- 좌표가 거래 신고 구·군 경계 밖: {int(mismatch.sum()):,} → 제외",
           f"- 좌표 미획득 단지 거래 수: {int(cx.loc[cx['lon'].isna(), 'n_trades'].sum()):,}"]
    ok = ok[~mismatch]

    # ---- 거리 변수 (m)
    sxy = to_xy(st["lon"], st["lat"])
    ok["dist_station"] = cKDTree(sxy).query(ok[["x", "y"]].values)[0]
    elem = sc[sc["category"].str.endswith("초등학교")]
    exy = to_xy(elem["lon"], elem["lat"])
    ok["dist_elem"] = cKDTree(exy).query(ok[["x", "y"]].values)[0]
    ok["dist_coast"] = gpd.GeoSeries(gpd.points_from_xy(ok.x, ok.y), crs=CRS).distance(sea).values
    for k, (lon, lat) in CBD.items():
        cxy = to_xy([lon], [lat])[0]
        ok[f"dist_{k}"] = np.hypot(ok.x - cxy[0], ok.y - cxy[1])
    ok["dist_cbd"] = ok[["dist_seomyeon", "dist_centum"]].min(axis=1)

    qa += ["", f"- 지하철역 {len(st)}개, 초등학교 {len(elem)}개 기준 최근접 거리 계산",
           f"- 단지→최근접 역 거리 중위값: {ok['dist_station'].median():,.0f} m"]
    with open(P("output", "geocoding_qa.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(qa) + "\n")
    print("\n".join(qa))

    keep = ["aptSeq", "lon", "lat", "x", "y", "dist_station", "dist_elem", "dist_coast",
            "dist_seomyeon", "dist_centum", "dist_cbd"]
    ok[keep].to_csv(P("data", "processed", "complex_features.csv"), index=False)
    tr = pd.read_parquet(P("data", "processed", "trades_clean.parquet"))
    tr = tr.merge(ok[keep], on="aptSeq", how="inner")
    tr.to_parquet(P("data", "processed", "trades_features.parquet"), index=False)
    print("trades with features:", len(tr))


if __name__ == "__main__":
    main()
