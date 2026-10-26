"""
04. 시각화 — 지도와 차트

출력: output/fig1_price_map.png, fig2_station_effect.png, fig3_price_index.png,
      fig4_residual_map.png, fig5_importance.png
"""
import glob
import json
import os
import warnings

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
CRS = 5179

INK = "#1F2933"
MUTED = "#6B7280"
SEA = "#EAF2F8"
LAND = "#F7F7F5"
ACCENT = "#C2410C"


def setup_font():
    for f in (glob.glob("/usr/share/fonts/**/NotoSansCJK-*.ttc", recursive=True)
              + glob.glob("/System/Library/Fonts/AppleSDGothicNeo.ttc")):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic"):
        if n in names:
            plt.rcParams["font.family"] = [n, "DejaVu Sans"]
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titleweight": "bold", "axes.titlesize": 13})


def base_map(ax, dist, sea, extent):
    ax.set_facecolor(SEA)
    dist.plot(ax=ax, color=LAND, edgecolor="#C9CED6", linewidth=0.6, zorder=1)
    sea.plot(ax=ax, color=SEA, linewidth=0, zorder=2)
    for _, r in dist.iterrows():
        c = r.geometry.representative_point()
        if extent[0] < c.x < extent[2] and extent[1] < c.y < extent[3]:
            ax.text(c.x, c.y, r["sggnm"], fontsize=7.5, color=MUTED, ha="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    setup_font()
    res = json.load(open(P("output", "results.json"), encoding="utf-8"))
    tr = pd.read_parquet(P("data", "processed", "trades_features.parquet"))
    tr = tr[~tr["direct"] & ~tr["outlier"]]
    dist = gpd.read_file(P("data", "geo", "busan_districts.gpkg")).to_crs(CRS)
    sea = gpd.read_file(P("data", "geo", "sea.gpkg")).to_crs(CRS)
    st = pd.read_csv(P("data", "raw", "busan_subway_stations_kakao.csv"))
    stg = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(CRS)
    cxy = tr.groupby("aptSeq")[["x", "y"]].first()
    pad = 2500
    extent = (cxy.x.quantile(0.002) - pad, cxy.y.quantile(0.002) - pad,
              cxy.x.quantile(0.998) + pad, cxy.y.quantile(0.998) + pad)

    # ---- 그림 1: 최근 15개월 단지별 ㎡당 중위가격
    recent = tr[tr["ym"] >= "2025-07"]
    cx = recent.groupby("aptSeq").agg(x=("x", "first"), y=("y", "first"),
                                      ppm2=("ppm2", "median"), n=("ppm2", "size")).reset_index()
    cx = cx.sort_values("ppm2")
    fig, ax = plt.subplots(figsize=(10, 9.2))
    base_map(ax, dist, sea, extent)
    stg.plot(ax=ax, marker="s", color=INK, markersize=5, zorder=4, alpha=0.7)
    vals = cx["ppm2"] / 1e4  # 만원/㎡
    sc = ax.scatter(cx.x, cx.y, c=vals, cmap="YlOrRd", s=np.clip(cx.n * 3, 8, 70),
                    vmin=np.percentile(vals, 3), vmax=np.percentile(vals, 97),
                    edgecolor="white", linewidth=0.3, zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01)
    cb.set_label("㎡당 중위 거래가 (만원)")
    ax.set_title(f"부산 아파트 단지별 ㎡당 거래가격 ({recent['ym'].min()} ~ {recent['ym'].max()})", loc="left")
    ax.text(0, -0.02, f"점 = 단지 {len(cx):,}곳 (크기 = 거래 건수) · ■ = 지하철역 · 출처: 국토교통부 실거래가, 카카오 로컬",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    fig.savefig(P("output", "fig1_price_map.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- 그림 2: 역세권 프리미엄 (모형 A, B)
    coef = pd.read_csv(P("output", "coefficients.csv"))
    bands = ["0-300m", "300-600m", "600-1000m", "1-2km"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.38
    for i, (m, col, lab) in enumerate([("A", "#9CA3AF", "모형 A (구 고정효과 없음)"),
                                        ("B", ACCENT, "모형 B (같은 구 안에서 비교)")]):
        c = coef[(coef.model == m) & coef.term.str.contains("station_band")].copy()
        c["band"] = c.term.str.extract(r"\[T\.(.+)\]")[0]
        c = c.set_index("band").loc[bands]
        xs = np.arange(len(bands)) + (i - 0.5) * w
        ax.bar(xs, c["pct"], width=w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, c["pct"], yerr=[c["pct"] - c["pct_lo"], c["pct_hi"] - c["pct"]],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=3)
        for x_, v in zip(xs, c["pct"]):
            ax.text(x_, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(bands)), [f"역까지 {b}" for b in bands])
    ax.set_ylabel("2km 이상 대비 ㎡당 가격 차이 (%)")
    ax.set_title("역세권 프리미엄: 면적·연식·층·시점을 통제한 가격 차이", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#E5E7EB", zorder=0)
    fig.text(0.01, -0.02, "오차막대 = 95% 신뢰구간 (단지 단위 군집 표준오차)", fontsize=8, color=MUTED)
    fig.savefig(P("output", "fig2_station_effect.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- 그림 3: 동일 단지 가격지수 vs 단순 중위가격
    idx = pd.read_csv(P("output", "price_index.csv"))
    med = tr.groupby("ym")["ppm2"].median()
    med = 100 * med / med.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = pd.to_datetime(idx["ym"])
    ax.plot(pd.to_datetime(med.index), med.values, color="#9CA3AF", lw=1.3,
            label="단순 ㎡당 중위가격 (거래되는 단지 구성에 따라 출렁임)")
    ax.plot(xs, idx["same_complex"], color=ACCENT, lw=2.4, label="동일 단지 가격지수 (단지 고정효과)")
    ax.axhline(100, color=MUTED, lw=0.7, ls=":")
    ax.set_ylabel("2021년 1월 = 100")
    ax.set_title("부산 아파트 가격 추이 (2021–2026)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color="#E5E7EB")
    last = idx["same_complex"].iloc[-1]
    ax.annotate(f"{last:.0f}", (xs.iloc[-1], last), xytext=(6, 0), textcoords="offset points",
                color=ACCENT, fontsize=9, va="center")
    fig.savefig(P("output", "fig3_price_index.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- 그림 4: 잔차 지도 (모형 A vs B)
    rr = pd.read_csv(P("data", "processed", "model_residuals.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, k, title in [(axes[0], "A", "모형 A 잔차"), (axes[1], "B", "모형 B 잔차 (구 고정효과 추가)")]:
        base_map(ax, dist, sea, extent)
        v = 100 * (np.exp(rr[f"res_{k}"]) - 1)
        s = ax.scatter(rr.x, rr.y, c=v, cmap="RdBu_r", vmin=-30, vmax=30, s=10,
                       edgecolor="none", zorder=5)
        I = res[f"model_{k}"]["morans_I"]
        ax.set_title(f"{title}   Moran's I = {I:.2f}", loc="left", fontsize=11.5)
    cb = fig.colorbar(s, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("예측 대비 실제 가격 (%)")
    fig.text(0.01, 0.0, "빨강 = 모형보다 비싸게 거래된 단지, 파랑 = 싸게 거래된 단지. Moran's I가 0에 가까울수록 남은 공간 패턴이 적음.",
             fontsize=8.5, color=MUTED)
    fig.savefig(P("output", "fig4_residual_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 그림 5: GBM 변수 중요도
    # '거래 시점'은 테스트 구간이 학습 구간 밖이라 트리 모형이 활용할 수 없음(중요도 0) → 그림에서 제외
    imp = pd.Series(res["gbm_importance"]).drop("t", errors="ignore").sort_values()
    names = {"log_area": "전용면적", "age": "건물 연식", "floor_c": "층", "dist_station": "역까지 거리",
             "dist_elem": "초등학교까지 거리", "dist_coast": "해안까지 거리", "dist_seomyeon": "서면까지 거리",
             "dist_centum": "센텀까지 거리", "lon": "경도", "lat": "위도", "t": "거래 시점"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([names.get(i, i) for i in imp.index], imp.values, color=ACCENT)
    ax.set_xlabel("순열 중요도 (테스트 구간 예측오차 증가량)")
    ax.set_title("머신러닝 모형이 가장 많이 쓴 변수", loc="left")
    ax.grid(axis="x", color="#E5E7EB")
    fig.savefig(P("output", "fig5_importance.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
