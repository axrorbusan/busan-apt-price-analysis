"""
03. 헤도닉 가격모형 + 공간 잔차 검정 + 머신러닝 비교

입력: data/processed/trades_features.parquet
출력: output/results.json, output/coefficients.csv, data/processed/model_residuals.csv
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)
RNG = np.random.default_rng(42)

STATION_BANDS = [0, 300, 600, 1000, 2000, np.inf]
STATION_LABELS = ["0-300m", "300-600m", "600-1000m", "1-2km", "2km+"]


def prepare(df):
    df = df[~df["direct"] & ~df["outlier"]].copy()
    df["log_ppm2"] = np.log(df["ppm2"])
    df["log_area"] = np.log(df["area"])
    df["age2"] = df["age"] ** 2 / 100
    df["floor_c"] = df["floor"].clip(upper=40)
    df["station_band"] = pd.cut(df["dist_station"], STATION_BANDS, labels=STATION_LABELS, right=False)
    df["log_coast"] = np.log(df["dist_coast"].clip(lower=50))
    df["log_cbd"] = np.log(df["dist_cbd"].clip(lower=200))
    df["elem_300"] = (df["dist_elem"] < 300).astype(int)
    return df


BASE = ("log_ppm2 ~ log_area + age + age2 + floor_c + "
        "C(station_band, Treatment('2km+')) + log_coast + log_cbd + elem_300 + C(ym)")


def morans_i(values, xy, k=8, perms=499):
    """kNN(k) 행표준화 가중치 Moran's I + 순열검정"""
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)
    nb = idx[:, 1:]
    z = values - values.mean()
    denom = (z ** 2).sum()
    def stat(zz):
        return (zz * zz[nb].mean(axis=1)).sum() / (zz ** 2).sum()
    I = stat(z)
    sims = np.array([stat(RNG.permutation(z)) for _ in range(perms)])
    p = (np.sum(sims >= I) + 1) / (perms + 1)
    return float(I), float(p)


def pct(b):
    return 100 * (np.exp(b) - 1)


def same_complex_index(df, months):
    """log(㎡당 가격) = 단지 고정효과 + 월 효과 + b1·log(면적) + b2·층  (희소행렬 최소제곱)"""
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    n = len(df)
    c = df["aptSeq"].astype("category").cat.codes.values
    m = pd.Categorical(df["ym"], categories=months).codes
    nc = c.max() + 1
    rows = np.arange(n)
    Xc = sparse.csr_matrix((np.ones(n), (rows, c)), shape=(n, nc))
    keep = m > 0                                            # 첫 달 = 기준
    Xm = sparse.csr_matrix((np.ones(keep.sum()), (rows[keep], m[keep] - 1)), shape=(n, len(months) - 1))
    Xo = sparse.csr_matrix(np.c_[df["log_area"].values, df["floor_c"].values])
    X = sparse.hstack([Xc, Xm, Xo]).tocsr()
    beta = lsqr(X, df["log_ppm2"].values, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    fe = beta[nc:nc + len(months) - 1]
    return [100.0] + list(100 * np.exp(fe))


def main():
    df = prepare(pd.read_parquet(P("data", "processed", "trades_features.parquet")))
    res = {"n_trades": int(len(df)), "n_complexes": int(df["aptSeq"].nunique()),
           "period": [df["ym"].min(), df["ym"].max()]}

    # ---- 모형 A: 입지 변수 + 월 고정효과 (구 고정효과 없음)
    # ---- 모형 B: + 구·군 고정효과 (같은 구 안에서의 입지 효과)
    models = {}
    for name, f in {"A": BASE, "B": BASE + " + C(sgg_nm)"}.items():
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["aptSeq"].astype("category").cat.codes})
        models[name] = m
        res[f"model_{name}"] = {"r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "n": int(m.nobs)}

    rows = []
    for name, m in models.items():
        ci = m.conf_int()
        for k in m.params.index:
            if k.startswith("C(ym)") or k == "Intercept":
                continue
            rows.append(dict(model=name, term=k, coef=m.params[k], se=m.bse[k], p=m.pvalues[k],
                             pct=pct(m.params[k]), pct_lo=pct(ci.loc[k, 0]), pct_hi=pct(ci.loc[k, 1])))
    coef = pd.DataFrame(rows)
    coef.to_csv(P("output", "coefficients.csv"), index=False, encoding="utf-8-sig")

    # 해석용 파생 수치 (모형 B)
    pb = models["B"].params
    res["derived_B"] = {
        "age_turning_point_years": float(-pb["age"] / (2 * pb["age2"] / 100)),
        "coast_distance_doubling_pct": float(100 * (2 ** pb["log_coast"] - 1)),
        "cbd_distance_doubling_pct": float(100 * (2 ** pb["log_cbd"] - 1)),
        "area_doubling_pct": float(100 * (2 ** pb["log_area"] - 1)),
        "per_floor_pct": float(pct(pb["floor_c"])),
        "elem_within_300m_pct": float(pct(pb["elem_300"])),
    }

    # 월별 가격지수 (2021-01 = 100)
    #  (1) 헤도닉 지수: 모형 B의 월 효과 — '같은 연식'의 아파트 기준 (노후화 효과 제거)
    #  (2) 동일 단지 지수: 단지 고정효과 + 월 효과 (면적·층 통제) — 같은 단지끼리 비교, 시장 지수에 가까움
    mb = models["B"]
    months = sorted(df["ym"].unique())
    hed = [100.0] + [100 * float(np.exp(mb.params.get(f"C(ym)[T.{m}]", np.nan))) for m in months[1:]]
    same = same_complex_index(df, months)
    pd.DataFrame({"ym": months, "hedonic": hed, "same_complex": same}).to_csv(
        P("output", "price_index.csv"), index=False)
    res["index_change_pct"] = {"same_complex": same[-1] - 100, "hedonic": hed[-1] - 100,
                               "peak_month_same_complex": months[int(np.argmax(same))],
                               "peak_same_complex": float(max(same))}

    # ---- 공간 잔차 검정: 단지별 평균 잔차의 Moran's I
    resid = df[["aptSeq", "x", "y", "lon", "lat", "sgg_nm"]].copy()
    resid["res_A"] = models["A"].resid
    resid["res_B"] = models["B"].resid
    cres = resid.groupby("aptSeq").agg(x=("x", "first"), y=("y", "first"), lon=("lon", "first"),
                                       lat=("lat", "first"), sgg_nm=("sgg_nm", "first"),
                                       res_A=("res_A", "mean"), res_B=("res_B", "mean"),
                                       n=("res_A", "size")).reset_index()
    cres = cres[cres["n"] >= 3]
    xy = cres[["x", "y"]].values
    for k in ("A", "B"):
        I, p = morans_i(cres[f"res_{k}"].values, xy)
        res[f"model_{k}"]["morans_I"] = I
        res[f"model_{k}"]["morans_p"] = p
    cres.to_csv(P("data", "processed", "model_residuals.csv"), index=False)

    # ---- 머신러닝 비교: 시간 기준 홀드아웃 (최근 15개월 = 테스트)
    feats = ["log_area", "age", "floor_c", "dist_station", "dist_elem", "dist_coast",
             "dist_seomyeon", "dist_centum", "lon", "lat", "t"]
    df["t"] = pd.to_datetime(df["ym"]).map(lambda d: (d.year - 2021) * 12 + d.month)
    cut = "2025-07"
    tr, te = df[df["ym"] < cut], df[df["ym"] >= cut]

    gbm = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    gbm.fit(tr[feats], tr["log_ppm2"])
    pred_g = gbm.predict(te[feats])

    # 같은 분할에서 OLS (월 효과 대신 선형 추세 — 미래 월은 학습에 없으므로)
    f_ols = BASE.replace("C(ym)", "t") + " + C(sgg_nm)"
    ols = smf.ols(f_ols, data=tr).fit()
    pred_o = ols.predict(te)

    def score(pred):
        e = te["log_ppm2"] - pred
        r2 = 1 - (e ** 2).sum() / ((te["log_ppm2"] - te["log_ppm2"].mean()) ** 2).sum()
        mape = float(np.median(np.abs(np.exp(e) - 1)))
        return {"r2": float(r2), "median_abs_pct_error": mape}
    res["holdout"] = {"train_until": cut, "n_train": int(len(tr)), "n_test": int(len(te)),
                      "ols": score(pred_o), "gbm": score(pred_g)}

    sample = te.sample(min(8000, len(te)), random_state=0)
    pi = permutation_importance(gbm, sample[feats], sample["log_ppm2"], n_repeats=5, random_state=0)
    res["gbm_importance"] = dict(sorted(zip(feats, map(float, pi.importances_mean)),
                                        key=lambda kv: -kv[1]))

    with open(P("output", "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(coef[coef.model == "B"][["term", "pct", "pct_lo", "pct_hi", "p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
