"""Pure statistical functions. Data-agnostic: no column names, no domain logic.

Every function is defensive -- on degenerate input it returns None (with the
caller recording a reason) rather than raising, so the pipeline never crashes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Information / diversity
# --------------------------------------------------------------------------- #
def shannon_entropy(counts) -> float | None:
    """Shannon entropy (nats) of a frequency vector."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size == 0:
        return None
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def normalized_entropy(counts) -> float | None:
    """Entropy scaled to [0,1] by log(n_categories). 1 = uniform, 0 = constant."""
    c = [x for x in counts if x > 0]
    if len(c) <= 1:
        return 0.0
    h = shannon_entropy(c)
    return None if h is None else float(h / np.log(len(c)))


def gini_impurity(counts) -> float | None:
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size == 0:
        return None
    p = c / c.sum()
    return float(1.0 - (p ** 2).sum())


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def outlier_bounds_iqr(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_fraction(s: pd.Series, method: str = "iqr", k: float = 1.5,
                     z: float = 3.0) -> dict:
    s = s.dropna()
    if s.empty:
        return {"fraction": None, "count": 0, "lower": None, "upper": None}
    if method == "iqr":
        lo, hi = outlier_bounds_iqr(s, k)
    elif method == "zscore":
        mu, sd = s.mean(), s.std()
        lo, hi = (mu - z * sd, mu + z * sd) if sd and sd > 0 else (s.min(), s.max())
    else:  # mad
        med = s.median()
        mad = (s - med).abs().median()
        scale = 1.4826 * mad
        lo, hi = (med - z * scale, med + z * scale) if scale > 0 else (s.min(), s.max())
    mask = (s < lo) | (s > hi)
    n = int(mask.sum())
    return {"fraction": n / len(s), "count": n, "lower": float(lo), "upper": float(hi)}


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float | None:
    """Population Stability Index between two samples of one variable.

    Numeric -> quantile bins from `expected`; categorical -> shared categories.
    """
    e, a = expected.dropna(), actual.dropna()
    if e.empty or a.empty:
        return None
    if pd.api.types.is_numeric_dtype(e):
        try:
            edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        except Exception:
            return None
        if edges.size < 3:
            return None
        edges[0], edges[-1] = -np.inf, np.inf
        e_pct = np.histogram(e, edges)[0] / len(e)
        a_pct = np.histogram(a, edges)[0] / len(a)
    else:
        cats = pd.Index(e.astype(str).unique()).union(a.astype(str).unique())
        e_pct = e.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy() / len(e)
        a_pct = a.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy() / len(a)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# --------------------------------------------------------------------------- #
# Association
# --------------------------------------------------------------------------- #
def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Bias-corrected Cramer's V between two categorical series."""
    tab = pd.crosstab(a, b)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    from scipy.stats import chi2_contingency
    try:
        chi2 = chi2_contingency(tab, correction=False)[0]
    except Exception:
        return None
    n = tab.to_numpy().sum()
    if n == 0:
        return None
    r, k = tab.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Eta: strength of a numeric variable's dependence on a categorical one."""
    df = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.empty or df["c"].nunique() < 2:
        return None
    grand = df["v"].mean()
    ss_between = df.groupby("c", observed=True)["v"].apply(
        lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    ss_total = ((df["v"] - grand) ** 2).sum()
    if ss_total == 0:
        return None
    return float(np.sqrt(ss_between / ss_total))


def conditional_entropy(x: pd.Series, y: pd.Series) -> float | None:
    """H(x | y) in nats -- expected residual uncertainty about x once y is known."""
    df = pd.DataFrame({"x": x.astype("object"), "y": y.astype("object")}).dropna()
    if df.empty:
        return None
    total = len(df)
    ent = 0.0
    for _, grp in df.groupby("y", observed=True)["x"]:
        vc = grp.value_counts().to_numpy(dtype=float)
        p = vc / vc.sum()
        ent += (len(grp) / total) * float(-(p * np.log(p)).sum())
    return ent


def theils_u(x: pd.Series, y: pd.Series) -> float | None:
    """Theil's U(x|y) in [0,1]: fraction of x's entropy explained by knowing y.
    Asymmetric -- U(x|y) != U(y|x) -- so it reveals *direction* of dependence,
    which symmetric Cramer's V cannot."""
    df = pd.DataFrame({"x": x.astype("object"), "y": y.astype("object")}).dropna()
    if df.empty or df["x"].nunique() < 2:
        return None
    hx = shannon_entropy(df["x"].value_counts().to_numpy())
    if not hx:
        return None
    hxy = conditional_entropy(df["x"], df["y"])
    if hxy is None:
        return None
    return float(max(0.0, min(1.0, (hx - hxy) / hx)))


def variance_inflation_factors(num: pd.DataFrame) -> dict | None:
    """VIF per column = diagonal of the inverse correlation matrix. Detects
    *multi-way* collinearity (a column linearly predictable from several others)
    that pairwise correlation misses. VIF>5 moderate, >10 severe."""
    X = num.dropna()
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 2 or len(X) < X.shape[1] + 2:
        return None
    cols = list(X.columns)
    corr = X.corr(method="pearson").to_numpy()
    if not np.isfinite(corr).all():
        return None
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    return {cols[i]: float(max(0.0, inv[i, i])) for i in range(len(cols))}


def decile_lift(x: pd.Series, y: pd.Series, bins: int = 10) -> dict | None:
    """Bin a numeric feature into quantiles and measure how concentrated the
    positives are. max_lift = highest-bin positive rate / base rate;
    top_bin_capture = share of all positives that fall in that bin."""
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y}).dropna()
    if df["y"].nunique() < 2 or df["x"].nunique() < 2:
        return None
    base = df["y"].mean()
    n_pos = df["y"].sum()
    if base == 0 or n_pos == 0:
        return None
    try:
        df["b"] = pd.qcut(df["x"], q=bins, duplicates="drop")
    except Exception:
        return None
    g = df.groupby("b", observed=True)["y"].agg(["mean", "sum"])
    top = g.sort_values("mean", ascending=False).iloc[0]
    return {"max_lift": float(g["mean"].max() / base),
            "top_bin_rate": float(top["mean"]),
            "top_bin_capture": float(top["sum"] / n_pos)}


def categorical_lift(x: pd.Series, y: pd.Series, min_n_frac: float = 0.01) -> dict | None:
    """Highest-risk category vs base rate. max_lift = max category positive rate /
    base rate (restricted to categories holding >=min_n_frac of rows)."""
    df = pd.DataFrame({"x": x.astype(str), "y": y}).dropna()
    if df["y"].nunique() < 2:
        return None
    base = df["y"].mean()
    if base == 0:
        return None
    g = df.groupby("x", observed=True)["y"].agg(["mean", "count"])
    g = g[g["count"] >= max(1, int(min_n_frac * len(df)))]
    if g.empty:
        return None
    return {"max_lift": float((g["mean"] / base).max()),
            "max_lift_category": str(g["mean"].idxmax()),
            "top_bad_rate": float(g["mean"].max())}


# --------------------------------------------------------------------------- #
# Target signal (supervised)
# --------------------------------------------------------------------------- #
def _woe_iv_from_table(good: np.ndarray, bad: np.ndarray):
    """WoE/IV from per-bin good (y=0) and bad (y=1) counts."""
    tot_good, tot_bad = good.sum(), bad.sum()
    if tot_good == 0 or tot_bad == 0:
        return None, None
    dist_good = np.clip(good / tot_good, 1e-6, None)
    dist_bad = np.clip(bad / tot_bad, 1e-6, None)
    woe = np.log(dist_bad / dist_good)
    iv = float(((dist_bad - dist_good) * woe).sum())
    return woe, iv


def iv_woe_categorical(x: pd.Series, y: pd.Series):
    df = pd.DataFrame({"x": x.astype("object").where(x.notna(), "__NA__"), "y": y}).dropna(subset=["y"])
    if df.empty or df["y"].nunique() < 2:
        return None, []
    g = df.groupby("x", observed=True)["y"]
    bad = g.sum().to_numpy(dtype=float)
    good = (g.count().to_numpy(dtype=float) - bad)
    woe, iv = _woe_iv_from_table(good, bad)
    if woe is None:
        return None, []
    cats = g.count().index.tolist()
    bins = [{"value": str(c), "n": int(good[i] + bad[i]), "bad": int(bad[i]),
             "bad_rate": float(bad[i] / (good[i] + bad[i])) if (good[i] + bad[i]) else None,
             "woe": float(woe[i])} for i, c in enumerate(cats)]
    return iv, bins


def iv_woe_numeric(x: pd.Series, y: pd.Series, max_bins: int = 10, min_frac: float = 0.05):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty or df["y"].nunique() < 2 or df["x"].nunique() < 2:
        return None, []
    try:
        df["bin"] = pd.qcut(df["x"], q=max_bins, duplicates="drop")
    except Exception:
        return None, []
    g = df.groupby("bin", observed=True)["y"]
    bad = g.sum().to_numpy(dtype=float)
    good = g.count().to_numpy(dtype=float) - bad
    woe, iv = _woe_iv_from_table(good, bad)
    if woe is None:
        return None, []
    bins = [{"interval": str(iv_), "n": int(good[i] + bad[i]), "bad": int(bad[i]),
             "bad_rate": float(bad[i] / (good[i] + bad[i])) if (good[i] + bad[i]) else None,
             "woe": float(woe[i])} for i, iv_ in enumerate(g.count().index)]
    return iv, bins


def iv_band(iv, weak=0.02, medium=0.1, strong=0.3) -> str | None:
    if iv is None:
        return None
    if iv < weak:
        return "useless"
    if iv < medium:
        return "weak"
    if iv < strong:
        return "medium"
    return "strong"


def single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df["y"].nunique() < 2 or df["x"].nunique() < 2:
        return None
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(df["y"], df["x"])
        return float(max(auc, 1 - auc))  # orientation-free
    except Exception:
        return None


def ks_statistic(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df["y"].nunique() < 2:
        return None
    from scipy.stats import ks_2samp
    pos = df.loc[df["y"] == 1, "x"]
    neg = df.loc[df["y"] == 0, "x"]
    if pos.empty or neg.empty:
        return None
    try:
        return float(ks_2samp(pos, neg).statistic)
    except Exception:
        return None


def point_biserial(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df["y"].nunique() < 2 or df["x"].nunique() < 2:
        return None
    from scipy.stats import pointbiserialr
    try:
        return float(pointbiserialr(df["y"], df["x"]).correlation)
    except Exception:
        return None
