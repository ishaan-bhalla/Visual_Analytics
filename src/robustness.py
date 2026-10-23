"""Robustness checks for the health-change models in src/analyse.py.

Reuses the same data, models and Bayesian fit, and asks how far the headline
numbers move when the data, the validation split or the prior change:

1. Leave-region-out refits (London, then each English region and Wales).
2. Spread of cross-validated R^2 over repeated 5-fold splits, and grouped
   cross-validation that holds out a whole region at a time.
3. Prior sensitivity: a shrinkage prior on the standardised slopes at several
   scales, against the reference prior used for the headline results.
4. Regression to the mean in the catch-up correlation.

Reads data/processed/areas_2011_2021.csv (from src/prepare.py) and writes
results/robustness.json. It does not touch results/summary.json.

Run from the repo root:  python src/robustness.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from analyse import SEED, bayes_fit, bayes_predict, cross_validate
from prepare import load

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MODELS = {
    "simple": ["unemployment_pct", "asian_pct"],
    "extended": ["good_or_very_good_health_pct", "unemployment_pct", "inactivity_pct",
                 "asian_pct", "moved_from_outside_uk_pct"],
}
UNEMP = "unemployment_pct"


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def unemployment_effect(m: pd.DataFrame, cols: list[str]) -> dict:
    """Per-SD unemployment coefficient (posterior mean and 95% CI)."""
    X = StandardScaler().fit_transform(m[cols])
    b = bayes_fit(X, m["good_change"].to_numpy())["beta"][:, 1 + cols.index(UNEMP)]
    return {"mean": round(float(b.mean()), 4), "ci_low": round(float(np.percentile(b, 2.5)), 4),
            "ci_high": round(float(np.percentile(b, 97.5)), 4)}


def fall_summary(m: pd.DataFrame) -> dict:
    """Simple vs extended unemployment effect, % fall and 5-fold CV R^2."""
    y = m["good_change"].to_numpy()
    simple = unemployment_effect(m, MODELS["simple"])
    extended = unemployment_effect(m, MODELS["extended"])
    return {
        "areas": int(len(m)),
        "unemployment_simple": simple,
        "unemployment_extended": extended,
        "fall_pct": round(100 * (1 - extended["mean"] / simple["mean"]), 1),
        "extended_ci_excludes_zero": bool(extended["ci_low"] > 0),
        "cv_r2_simple": cross_validate(m[MODELS["simple"]].to_numpy(), y)["r2"],
        "cv_r2_extended": cross_validate(m[MODELS["extended"]].to_numpy(), y)["r2"],
    }


def bootstrap_fall(m: pd.DataFrame, reps: int = 2000) -> dict:
    """Resample areas and recompute the % fall (posterior means = least squares
    under the reference prior), to put an interval on the 60% figure."""
    rng = np.random.default_rng(SEED)
    y = m["good_change"].to_numpy()
    Xs, Xe = m[MODELS["simple"]].to_numpy(), m[MODELS["extended"]].to_numpy()
    falls = []
    for _ in range(reps):
        i = rng.integers(0, len(m), len(m))
        coefs = []
        for X, cols in [(Xs, MODELS["simple"]), (Xe, MODELS["extended"])]:
            Z = StandardScaler().fit_transform(X[i])
            Zb = np.hstack([np.ones((len(i), 1)), Z])
            coefs.append(np.linalg.lstsq(Zb, y[i], rcond=None)[0][1 + cols.index(UNEMP)])
        falls.append(100 * (1 - coefs[1] / coefs[0]))
    falls = np.array(falls)
    return {"reps": reps, "median_pct": round(float(np.median(falls)), 1),
            "ci_low_pct": round(float(np.percentile(falls, 2.5)), 1),
            "ci_high_pct": round(float(np.percentile(falls, 97.5)), 1),
            "share_fall_above_50pct": round(float((falls > 50).mean()), 3)}


def cv_predictions(X: np.ndarray, y: np.ndarray, splits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, lo, hi = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    for train, test in splits:
        sc = StandardScaler().fit(X[train])
        post = bayes_fit(sc.transform(X[train]), y[train])
        pred[test], lo[test], hi[test] = bayes_predict(post, sc.transform(X[test]))
    return pred, lo, hi


def repeated_cv(m: pd.DataFrame, cols: list[str], repeats: int = 20, folds: int = 5) -> dict:
    X, y = m[cols].to_numpy(), m["good_change"].to_numpy()
    scores, covers = [], []
    for r in range(repeats):
        pred, lo, hi = cv_predictions(X, y, KFold(folds, shuffle=True, random_state=r).split(X))
        scores.append(r2(y, pred))
        covers.append(float(((y >= lo) & (y <= hi)).mean()))
    s = np.array(scores)
    return {"repeats": repeats, "folds": folds, "r2_mean": round(float(s.mean()), 3),
            "r2_sd": round(float(s.std(ddof=1)), 3), "r2_min": round(float(s.min()), 3),
            "r2_max": round(float(s.max()), 3), "coverage_mean": round(float(np.mean(covers)), 3)}


def region_cv(m: pd.DataFrame, cols: list[str]) -> dict:
    """Train on nine regions, predict the tenth; repeat for every region."""
    X, y, g = m[cols].to_numpy(), m["good_change"].to_numpy(), m["region"].to_numpy()
    pred, lo, hi = cv_predictions(X, y, LeaveOneGroupOut().split(X, y, g))
    per_region = []
    for reg in sorted(set(g)):
        k = g == reg
        per_region.append({"region": reg, "areas": int(k.sum()),
                           "rmse_pp": round(float(np.sqrt(((y[k] - pred[k]) ** 2).mean())), 3),
                           "mean_error_pp": round(float((y[k] - pred[k]).mean()), 3),
                           "coverage_95": round(float(((y[k] >= lo[k]) & (y[k] <= hi[k])).mean()), 3)})
    return {"r2": round(r2(y, pred), 3),
            "rmse_pp": round(float(np.sqrt(((y - pred) ** 2).mean())), 3),
            "coverage_95": round(float(((y >= lo) & (y <= hi)).mean()), 3),
            "per_region": per_region}


def shrinkage_fit(X: np.ndarray, y: np.ndarray, tau: float, draws: int = 4000) -> dict:
    """Conjugate prior beta_j | sigma^2 ~ N(0, tau^2 sigma^2) on each standardised
    slope, flat on the intercept, p(sigma^2) ~ 1/sigma^2. Smaller tau shrinks
    the slopes harder towards zero; tau -> infinity recovers the reference prior."""
    rng = np.random.default_rng(SEED)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    n, p = Xb.shape
    prec = np.diag([0.0] + [1 / tau ** 2] * (p - 1))
    V = np.linalg.inv(Xb.T @ Xb + prec)
    b = V @ Xb.T @ y
    s = float(y @ y - b @ (Xb.T @ Xb + prec) @ b)
    sigma2 = s / rng.chisquare(n - 1, size=draws)
    L = np.linalg.cholesky(V)
    beta = b + (rng.standard_normal((draws, p)) @ L.T) * np.sqrt(sigma2)[:, None]
    return {"beta": beta, "sigma2": sigma2}


def prior_sensitivity(m: pd.DataFrame, taus=(0.1, 0.25, 0.5, 1.0, 10.0)) -> list[dict]:
    y = m["good_change"].to_numpy()
    rows = []
    for tau in [None, *taus]:
        fit = bayes_fit if tau is None else (lambda X, yy, t=tau: shrinkage_fit(X, yy, t))
        entry = {"prior": "reference (headline)" if tau is None else f"N(0, ({tau} sigma)^2) on slopes"}
        for name, cols in MODELS.items():
            X = m[cols].to_numpy()
            post = fit(StandardScaler().fit_transform(X), y)
            coefs = {}
            for j, c in enumerate(cols):
                bj = post["beta"][:, 1 + j]
                coefs[c] = {"mean": round(float(bj.mean()), 4),
                            "ci_low": round(float(np.percentile(bj, 2.5)), 4),
                            "ci_high": round(float(np.percentile(bj, 97.5)), 4)}
            pred, lo, hi = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
            for train, test in KFold(5, shuffle=True, random_state=SEED).split(X):
                sc = StandardScaler().fit(X[train])
                p_ = fit(sc.transform(X[train]), y[train])
                pred[test], lo[test], hi[test] = bayes_predict(p_, sc.transform(X[test]))
            entry[name] = {"coefficients_per_sd": coefs, "cv_r2": round(r2(y, pred), 3),
                           "cv_coverage_95": round(float(((y >= lo) & (y <= hi)).mean()), 3)}
        s, e = entry["simple"]["coefficients_per_sd"][UNEMP], entry["extended"]["coefficients_per_sd"][UNEMP]
        entry["unemployment_fall_pct"] = round(100 * (1 - e["mean"] / s["mean"]), 1)
        rows.append(entry)
    return rows


def regression_to_mean(m: pd.DataFrame) -> dict:
    """How much of the catch-up correlation could pure sampling noise produce?

    If an area's true good-health share were unchanged and each Census added
    independent noise e (variance s2), then corr(h2011, change) would be
    negative purely from noise: cov(h2011, h2021 - h2011) = -s2. The Census is a
    near-complete count, so the binomial noise of a share p in N residents,
    p(1-p)/N, is used as the sampling-noise variance. Reporting noise (how
    people rate their health on the day) is not measurable from this data.
    """
    h = load("health.csv", {"disability/health/care: All categories: Long-term health problem "
                            "or disability; measures: Value": "residents"})[["area_code", "residents"]]
    d = m.merge(h, on="area_code", how="left")
    p = d["good_or_very_good_health_pct"].to_numpy() / 100
    n = d["residents"].to_numpy()
    h11 = d["good_or_very_good_health_pct"].to_numpy()
    change = d["good_change"].to_numpy()
    noise_var = float(np.mean(p * (1 - p) / n) * 100 ** 2)  # pp^2
    var_h11, var_change = float(h11.var(ddof=1)), float(change.var(ddof=1))
    obs_cov = float(np.cov(h11, change)[0, 1])
    obs_r = float(np.corrcoef(h11, change)[0, 1])
    # Expected corr from sampling noise alone: -s2 / sqrt(var_h11 * var_change)
    rtm_r_sampling = -noise_var / np.sqrt(var_h11 * var_change)
    # Noise variance that would be needed for RTM to explain all the catch-up,
    # and the reliability of the 2011 measure that this implies.
    needed = -obs_cov
    # Alternative baseline: catch-up measured against a different 2011 health
    # question (day-to-day activities limited), which does not share the
    # good-health item's own question-specific error.
    alt_r = float(np.corrcoef(d["limited_any_pct"], change)[0, 1])
    return {
        "observed_r": round(obs_r, 3),
        "observed_slope_pp_per_pp": round(obs_cov / var_h11, 4),
        "sampling_noise_sd_pp": round(float(np.sqrt(noise_var)), 4),
        "var_2011_good_health_pp2": round(var_h11, 3),
        "expected_r_from_sampling_noise": round(float(rtm_r_sampling), 5),
        "noise_sd_needed_to_explain_all_pp": round(float(np.sqrt(needed)), 3) if needed > 0 else None,
        "implied_reliability_if_all_rtm": round(1 - needed / var_h11, 3),
        "r_change_vs_2011_limited_activities": round(alt_r, 3),
        "corr_2011_good_health_vs_limited": round(float(np.corrcoef(h11, d["limited_any_pct"])[0, 1]), 3),
    }


def main() -> None:
    df = pd.read_csv(DATA / "areas_2011_2021.csv")
    m = df[df.matched_2021].reset_index(drop=True)
    out: dict = {"areas": int(len(m))}

    # 1. Leave-region-out.
    out["full_sample"] = fall_summary(m)
    out["bootstrap_fall"] = bootstrap_fall(m)
    out["leave_region_out"] = [{"excluded": reg, **fall_summary(m[m.region != reg].reset_index(drop=True))}
                               for reg in sorted(m.region.unique())]

    # 2. Cross-validation spread and region-grouped CV.
    out["repeated_cv"] = {name: repeated_cv(m, cols) for name, cols in MODELS.items()}
    out["region_grouped_cv"] = {name: region_cv(m, cols) for name, cols in MODELS.items()}

    # 3. Prior sensitivity.
    out["prior_sensitivity"] = prior_sensitivity(m)

    # 4. Regression to the mean in the catch-up correlation.
    out["regression_to_mean"] = regression_to_mean(m)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
