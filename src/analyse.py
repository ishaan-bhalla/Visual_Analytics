"""PCA, t-SNE, clustering and Bayesian regression on the prepared areas.

Writes per-area results to data/processed/analysis.csv and every headline
number to results/summary.json, which the README and dashboard read from.

Run from the repo root:  python src/analyse.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SEED = 42

SOCIO = ["employment_pct", "unemployment_pct", "inactivity_pct", "full_time_pct", "part_time_pct",
         "white_pct", "mixed_pct", "asian_pct", "black_pct", "other_pct",
         "same_address_pct", "moved_within_area_pct", "moved_from_uk_pct",
         "moved_from_outside_uk_pct", "moved_out_pct"]
HEALTH_2011 = ["limited_any_pct", "not_limited_pct", "good_or_very_good_health_pct",
               "fair_health_pct", "bad_or_very_bad_health_pct"]
ALL_2011 = SOCIO + HEALTH_2011  # the 20 features the marked version used


def pca(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    model = PCA(n_components=2, random_state=SEED).fit(StandardScaler().fit_transform(df[cols]))
    scores = model.transform(StandardScaler().fit_transform(df[cols]))
    loadings = pd.DataFrame(model.components_.T, index=cols, columns=["PC1", "PC2"])
    top = {pc: loadings[pc].abs().sort_values(ascending=False).head(5).index.tolist() for pc in ["PC1", "PC2"]}
    info = {
        "features": len(cols),
        "explained_variance": [round(float(v), 4) for v in model.explained_variance_ratio_],
        "explained_total": round(float(model.explained_variance_ratio_.sum()), 4),
        "top_loadings": {pc: {c: round(float(loadings.loc[c, pc]), 3) for c in top[pc]} for pc in top},
    }
    return scores, info


def clusters(X: np.ndarray) -> tuple[np.ndarray, dict]:
    scores = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=SEED).fit_predict(X)) for k in range(2, 9)}
    best = max(scores, key=scores.get)
    labels = KMeans(best, n_init=20, random_state=SEED).fit_predict(X)
    return labels, {"k": best, "silhouette": {k: round(float(v), 3) for k, v in scores.items()}}


def name_clusters(df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.Series, list[dict]]:
    """Name each cluster from where its averages sit relative to all areas."""
    df = df.assign(cluster=labels)
    z = (df.groupby("cluster")[SOCIO + ["good_or_very_good_health_pct"]].mean() - df[SOCIO + ["good_or_very_good_health_pct"]].mean()) \
        / df[SOCIO + ["good_or_very_good_health_pct"]].std()
    names, profiles = {}, []
    for c, row in z.iterrows():
        diverse = row[["asian_pct", "black_pct", "mixed_pct", "other_pct"]].mean()
        mobile = row[["moved_from_outside_uk_pct", "moved_within_area_pct", "moved_from_uk_pct"]].mean()
        if diverse > 1:
            name = "Diverse, high-migration urban areas" if mobile > 0.5 else "Diverse urban areas"
        elif row["unemployment_pct"] > 0.5 or row["inactivity_pct"] > 0.5:
            name = "Higher unemployment and inactivity"
        elif row["employment_pct"] > 0.3 and row["unemployment_pct"] < 0:
            name = "Employed, settled areas"
        elif diverse < 0 and mobile < 0:
            name = "Less diverse, lower-migration areas"
        else:
            name = "Average areas"
        names[c] = name
        means = df[df.cluster == c]
        profiles.append({
            "cluster": int(c), "name": name, "areas": int(len(means)),
            "unemployment_pct": round(float(means.unemployment_pct.mean()), 2),
            "white_pct": round(float(means.white_pct.mean()), 2),
            "moved_from_outside_uk_pct": round(float(means.moved_from_outside_uk_pct.mean()), 2),
            "good_health_2011_pct": round(float(means.good_or_very_good_health_pct.mean()), 2),
            "good_change_pp": round(float(means.good_change.mean()), 3),
        })
    # Disambiguate duplicate names with their size order.
    seen: dict[str, int] = {}
    for p in sorted(profiles, key=lambda p: -p["areas"]):
        seen[p["name"]] = seen.get(p["name"], 0) + 1
        if seen[p["name"]] > 1:
            p["name"] = f"{p['name']} ({seen[p['name']]})"
            names[p["cluster"]] = p["name"]
    return pd.Series(labels).map(names), profiles


def bayes_fit(X: np.ndarray, y: np.ndarray, draws: int = 4000) -> dict:
    """Conjugate Bayesian linear regression with the reference prior
    p(beta, sigma^2) ~ 1/sigma^2, as in the marked version. Returns posterior draws."""
    rng = np.random.default_rng(SEED)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    n, p = Xb.shape
    xtx_inv = np.linalg.inv(Xb.T @ Xb)
    beta_hat = xtx_inv @ Xb.T @ y
    rss = float(((y - Xb @ beta_hat) ** 2).sum())
    sigma2 = rss / rng.chisquare(n - p, size=draws)
    L = np.linalg.cholesky(xtx_inv)
    beta = beta_hat + (rng.standard_normal((draws, p)) @ L.T) * np.sqrt(sigma2)[:, None]
    return {"beta": beta, "sigma2": sigma2}


def bayes_predict(post: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive mean and 95% interval (includes observation noise)."""
    rng = np.random.default_rng(SEED + 1)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    mu = post["beta"] @ Xb.T
    sims = mu + rng.standard_normal(mu.shape) * np.sqrt(post["sigma2"])[:, None]
    return mu.mean(0), np.percentile(sims, 2.5, 0), np.percentile(sims, 97.5, 0)


def coef_table(post: dict, names: list[str]) -> list[dict]:
    rows = []
    for i, name in enumerate(["intercept"] + names):
        b = post["beta"][:, i]
        rows.append({"term": name, "mean": round(float(b.mean()), 4),
                     "ci_low": round(float(np.percentile(b, 2.5)), 4),
                     "ci_high": round(float(np.percentile(b, 97.5)), 4),
                     "p_positive": round(float((b > 0).mean()), 3)})
    return rows


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    """Out-of-sample RMSE, R^2 and 95% interval coverage over K folds."""
    pred, lo, hi = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    for train, test in KFold(folds, shuffle=True, random_state=SEED).split(X):
        sc = StandardScaler().fit(X[train]) if X.shape[1] else None
        Xtr = sc.transform(X[train]) if sc else X[train]
        Xte = sc.transform(X[test]) if sc else X[test]
        post = bayes_fit(Xtr, y[train])
        pred[test], lo[test], hi[test] = bayes_predict(post, Xte)
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    cover = float(((y >= lo) & (y <= hi)).mean())
    return {"rmse_pp": round(rmse, 3), "r2": round(r2, 3), "coverage_95": round(cover, 3)}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "areas_2011_2021.csv")
    summary: dict = {"areas_2011": int(len(df)), "areas_matched_2021": int(df.matched_2021.sum())}

    # 1. PCA as marked (20 features incl. health) and socio-economic only (15).
    pca_all, summary["pca_marked_20"] = pca(df, ALL_2011)
    pca_soc, summary["pca_socio_15"] = pca(df, SOCIO)
    df[["pc1_all", "pc2_all"]] = pca_all
    df[["pc1", "pc2"]] = pca_soc

    # 2. t-SNE on the socio-economic features.
    Xs = StandardScaler().fit_transform(df[SOCIO])
    df[["tsne1", "tsne2"]] = TSNE(2, perplexity=30, random_state=SEED, init="pca").fit_transform(Xs)

    # 3. Clusters on the socio-economic features (k chosen by silhouette).
    labels, summary["clustering"] = clusters(Xs)
    df["cluster"], summary["clusters"] = name_clusters(df, labels)

    # 4. Bayesian regression of 2011-21 change in good health (323 matched areas).
    m = df[df.matched_2021].reset_index(drop=True)
    y = m["good_change"].to_numpy()
    models = {
        "intercept_only": [],
        "marked_version": ["unemployment_pct", "asian_pct"],
        "extended": ["good_or_very_good_health_pct", "unemployment_pct", "inactivity_pct",
                     "asian_pct", "moved_from_outside_uk_pct"],
    }
    summary["models"] = {}
    for name, cols in models.items():
        X = m[cols].to_numpy() if cols else np.zeros((len(m), 0))
        entry = {"predictors": cols, "cv_5fold": cross_validate(X, y)}
        if name == "marked_version":
            # Raw (unscaled) coefficients, comparable with the marked report.
            entry["coefficients_raw"] = coef_table(bayes_fit(X, y), cols)
        if cols:
            sc = StandardScaler().fit(X)
            post = bayes_fit(sc.transform(X), y)
            entry["coefficients_per_sd"] = coef_table(post, cols)
            if name == "extended":
                mean, lo, hi = bayes_predict(post, sc.transform(X))
                m["predicted_change"], m["pi_low"], m["pi_high"] = mean, lo, hi
                m["residual"] = y - mean
                entry["in_sample_r2"] = round(float(1 - ((y - mean) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)
        summary["models"][name] = entry

    df = df.merge(m[["area_code", "predicted_change", "pi_low", "pi_high", "residual"]], on="area_code", how="left")
    summary["good_change"] = {
        "mean_pp": round(float(y.mean()), 3), "min_pp": round(float(y.min()), 2), "max_pp": round(float(y.max()), 2),
        "areas_improved": int((y > 0).sum()), "areas_worsened": int((y < 0).sum()),
        "corr_with_2011_health": round(float(np.corrcoef(m.good_or_very_good_health_pct, y)[0, 1]), 3),
    }
    summary["by_region"] = (m.groupby("region")
                            .agg(areas=("area", "size"),
                                 good_2011=("good_or_very_good_health_pct", "mean"),
                                 good_2021=("good_or_very_good_health_pct_2021", "mean"),
                                 change=("good_change", "mean"))
                            .round(3).reset_index().to_dict("records"))

    df.round(4).to_csv(DATA / "analysis.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["pca_marked_20", "pca_socio_15", "clustering", "clusters",
                                              "models", "good_change"]}, indent=1))


if __name__ == "__main__":
    main()
