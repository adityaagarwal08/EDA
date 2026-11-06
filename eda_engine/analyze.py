"""Stage 2 -- ANALYZE: everything computed per column / per entity --
semantic typing, univariate profiling, the data-quality scorecard + remediation
actions, within-entity relationships (correlation / association / functional
dependencies), and a leakage-safe train/valid time-split suggestion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import metrics, io
from .setup import column_override
from .utils.typing_registry import (infer_semantic_type, family,
                              RE_IPV4, RE_IPV6, RE_EMAIL, RE_VPA, RE_IFSC)



# ============================================================================
# Semantic typing
# ============================================================================

def type_entity(df: pd.DataFrame, ecfg: dict, thresholds: dict) -> dict:
    pk = set(ecfg.get("primary_key") or [])
    fks = set((ecfg.get("foreign_keys") or {}).keys())
    out = {}
    for col in df.columns:
        ov = column_override(ecfg, col)
        is_key = col in pk or col in fks
        if "semantic_type" in ov:
            st = {"semantic_type": ov["semantic_type"], "confidence": 1.0,
                  "evidence": "config override", "source": "override"}
        else:
            st = {**infer_semantic_type(df[col], thresholds, is_key=is_key),
                  "source": "inferred"}
        # Coercion plan + parse-fail rate (fail-soft type downgrade).
        coerced, fail = io.coerce(df[col], st["semantic_type"])
        if fail > 0.5 and st["source"] == "inferred":
            # downgrade noisy numeric/datetime guesses to categorical/text
            st["semantic_type"] = "categorical" if df[col].nunique() <= \
                thresholds["type_inference"]["categorical_max_unique"] else "free_text"
            st["evidence"] += f"; downgraded (parse_fail={fail:.2f})"
            coerced, fail = io.coerce(df[col], st["semantic_type"])
        st.update({
            "physical_dtype": str(df[col].dtype),
            "family": family(st["semantic_type"]),
            "parse_fail_rate": round(fail, 4),
            "is_primary_key": col in pk,
            "is_foreign_key": col in fks,
            "feature_role": "key" if (col in pk or col in fks or
                                      st["semantic_type"] in ("identifier", "hashed_pii"))
            else "feature",
        })
        out[col] = st
    return out



# ============================================================================
# Univariate profiling
# ============================================================================

def _universal(s: pd.Series, thresholds: dict, top_k: int) -> dict:
    n = len(s)
    n_miss = int(s.isna().sum())
    nonnull = s.dropna()
    n_present = len(nonnull)
    n_unique = int(nonnull.nunique())
    vc = nonnull.astype("object").value_counts()
    top = [{"value": str(k), "count": int(v), "pct": float(v / n_present)}
           for k, v in vc.head(top_k).items()] if n_present else []
    dominance = float(vc.iloc[0] / n_present) if n_present else None
    as_str = nonnull.astype(str)
    blanks = int((as_str.str.strip() == "").sum())
    return {
        "count": n, "n_present": n_present, "n_missing": n_miss,
        "pct_missing": round(n_miss / n, 6) if n else None,
        "n_unique": n_unique,
        "cardinality_ratio": round(n_unique / n_present, 6) if n_present else None,
        "memory_bytes": int(s.memory_usage(deep=True)),
        "n_blank": blanks,
        "top_values": top,
        "dominance": dominance,
        "is_constant": n_unique <= 1,
        "is_near_constant": bool(dominance is not None and
                                 dominance >= thresholds["quality"]["near_constant_dominance"]),
        "entropy": metrics.shannon_entropy(vc.to_numpy()) if n_present else None,
        "normalized_entropy": metrics.normalized_entropy(vc.to_numpy()) if n_present else None,
    }


def _profile_numeric(s: pd.Series, thresholds: dict) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"_note": "no numeric values"}
    q = s.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_dict()
    qa = thresholds["quality"]
    out = {
        "min": float(s.min()), "max": float(s.max()), "range": float(s.max() - s.min()),
        "mean": float(s.mean()), "trimmed_mean_10": float(s[(s >= s.quantile(.05)) &
                                                            (s <= s.quantile(.95))].mean()),
        "median": float(s.median()), "std": float(s.std()), "var": float(s.var()),
        "sum": float(s.sum()),
        "mad": float((s - s.median()).abs().median()),
        "iqr": float(s.quantile(.75) - s.quantile(.25)),
        "cv": float(s.std() / s.mean()) if s.mean() else None,
        "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
        "percentiles": {f"p{int(k*100)}": float(v) for k, v in q.items()},
        "n_zero": int((s == 0).sum()), "n_negative": int((s < 0).sum()),
        "zero_inflation": float((s == 0).mean()),
        "all_positive": bool((s > 0).all()),
        "is_integer_valued": bool(np.array_equal(s, s.round())),
        "outliers": metrics.outlier_fraction(s, qa["outlier_method"],
                                              qa["outlier_iqr_k"], qa["outlier_zscore"]),
    }
    # distribution-shape tag + transform hint
    sk = out["skew"]
    out["shape"] = ("right_skewed" if sk > 1 else "left_skewed" if sk < -1
                    else "approx_symmetric")
    out["log_transform_candidate"] = bool(out["all_positive"] and sk > 1)
    # monotonicity in row order (flags running totals / pre-sorted columns)
    out["is_monotonic"] = ("increasing" if s.is_monotonic_increasing
                           else "decreasing" if s.is_monotonic_decreasing else "none")
    # histogram for report
    hist, edges = np.histogram(s, bins=thresholds["report"]["hist_bins"])
    out["histogram"] = {"counts": hist.tolist(), "edges": [float(e) for e in edges]}
    return out


def _profile_categorical(s: pd.Series, thresholds: dict) -> dict:
    nonnull = s.dropna().astype("object")
    if nonnull.empty:
        return {"_note": "empty"}
    vc = nonnull.value_counts()
    n = len(nonnull)
    min_freq = max(1, int(0.01 * n))
    rare = vc[vc < min_freq]
    as_str = nonnull.astype(str)
    # case/whitespace variant detection
    norm = as_str.str.strip().str.lower()
    variant_collapse = int(as_str.nunique() - norm.nunique())
    return {
        "n_levels": int(vc.size),
        "max_class_share": float(vc.iloc[0] / n),
        "imbalance_ratio": float(vc.iloc[0] / vc.iloc[-1]) if vc.size > 1 else None,
        "gini_impurity": metrics.gini_impurity(vc.to_numpy()),
        "n_rare_levels": int(rare.size),
        "rare_level_mass": float(rare.sum() / n) if rare.size else 0.0,
        "case_whitespace_variants": variant_collapse,
        "imbalanced": bool(vc.iloc[0] / n > 0.95),
    }


def _profile_boolean(s: pd.Series, thresholds: dict) -> dict:
    c, _ = io.coerce(s, "boolean_flag")
    c = pd.to_numeric(c, errors="coerce").dropna()
    if c.empty:
        return {"_note": "uncoercible"}
    return {"true_rate": float((c == 1).mean()), "n_true": int((c == 1).sum()),
            "n_false": int((c == 0).sum())}


def _profile_ordinal(s: pd.Series, order, thresholds: dict) -> dict:
    base = _profile_categorical(s, thresholds)
    if order:
        present = set(s.dropna().astype(str))
        order_str = [str(o) for o in order]
        base["order"] = order_str
        base["missing_levels"] = [o for o in order_str if o not in present]
        base["unexpected_levels"] = [str(v) for v in present if v not in set(order_str)]
    return base


def _profile_datetime(s: pd.Series, ecfg: dict, col: str, thresholds: dict) -> dict:
    d, fail = io.coerce(s, "datetime")
    d = pd.to_datetime(d, errors="coerce").dropna()
    if d.empty:
        return {"_note": "no parseable dates", "parse_fail_rate": fail}
    now = pd.Timestamp.utcnow().tz_localize(None)
    sentinels = {"1900-01-01", "1970-01-01", "1899-12-30"}
    n_sent = int(d.dt.strftime("%Y-%m-%d").isin(sentinels).sum())
    has_time = bool((d.dt.hour.nunique() > 1) or (d.dt.minute.nunique() > 1))
    # largest gap between consecutive distinct days -> coverage holes in the timeline
    uniq_days = d.dt.normalize().drop_duplicates().sort_values()
    max_gap = int(uniq_days.diff().dt.days.max()) if len(uniq_days) > 1 else 0
    out = {
        "parse_fail_rate": round(fail, 4),
        "min": d.min().isoformat(), "max": d.max().isoformat(),
        "span_days": int((d.max() - d.min()).days),
        "n_unique_days": int(d.dt.normalize().nunique()),
        "max_gap_days": max_gap,
        "is_monotonic_increasing": bool(d.is_monotonic_increasing),
        "has_time_component": has_time,
        "n_future_dates": int((d > now).sum()),
        "n_sentinel_dates": n_sent,
        "by_year": {str(k): int(v) for k, v in d.dt.year.value_counts().sort_index().items()},
        "by_dow": {str(k): int(v) for k, v in d.dt.dayofweek.value_counts().sort_index().items()},
    }
    if has_time:
        out["by_hour"] = {str(k): int(v) for k, v in d.dt.hour.value_counts().sort_index().items()}
    # derived-feature stubs
    derived = ["recency_days", "year", "month", "dayofweek"]
    if has_time:
        derived.append("hour")
    if column_is_dob(ecfg, col):
        derived.append("age_years")
    out["derived_feature_stubs"] = derived
    return out


def column_is_dob(ecfg: dict, col: str) -> bool:
    ov = (ecfg.get("column_overrides") or {}).get(col, {}) or {}
    return bool(ov.get("derive_age"))


_PATTERN_RE = {"ip_address": lambda s: bool(RE_IPV4.match(s) or RE_IPV6.match(s)),
               "email": lambda s: bool(RE_EMAIL.match(s)),
               "vpa": lambda s: bool(RE_VPA.match(s)),
               "bank_code": lambda s: bool(RE_IFSC.match(s))}


def _profile_pattern(s: pd.Series, st: str, thresholds: dict) -> dict:
    nonnull = s.dropna().astype(str).str.strip()
    if nonnull.empty:
        return {"_note": "empty"}
    out = {"length_min": int(nonnull.str.len().min()),
           "length_max": int(nonnull.str.len().max()),
           "length_mode": int(nonnull.str.len().mode().iloc[0])}
    if st in _PATTERN_RE:
        valid = nonnull.map(_PATTERN_RE[st])
        out["validity_rate"] = float(valid.mean())
        bad = nonnull[~valid]
        out["invalid_examples"] = bad.head(thresholds["report"]["max_invalid_examples"]).tolist()
    elif st == "pincode":
        valid = nonnull.str.fullmatch(r"\d{6}")
        out["validity_rate"] = float(valid.mean())
    elif st == "phone":
        digits = nonnull.str.replace(r"\D", "", regex=True)
        out["validity_rate"] = float(digits.str.len().between(10, 12).mean())
    # decomposition stubs for FE
    stubs = {"ip_address": ["subnet_24", "is_private"],
             "vpa": ["psp_handle", "known_psp_flag"],
             "bank_code": ["bank_prefix", "branch_code"],
             "email": ["domain"], "pincode": ["region_lookup(requires_human)"],
             "phone": ["valid_flag"]}.get(st, [])
    out["decomposition_stubs"] = stubs
    return out


def _profile_text(s: pd.Series, thresholds: dict) -> dict:
    nonnull = s.dropna().astype(str)
    if nonnull.empty:
        return {"_note": "empty"}
    samp = nonnull
    cap = thresholds["report"].get("top_k_values", 20)
    lens = nonnull.str.len()
    toks = nonnull.str.split()
    return {
        "char_len_mean": float(lens.mean()), "char_len_max": int(lens.max()),
        "token_count_mean": float(toks.map(len).mean()),
        "whitespace_only_frac": float((nonnull.str.strip() == "").mean()),
        "embedded_digit_frac": float(nonnull.str.contains(r"\d").mean()),
        "note": "vectorization deferred (requires_human / later stage)",
    }


def _profile_keylike(s: pd.Series, thresholds: dict) -> dict:
    nonnull = s.dropna().astype(str)
    if nonnull.empty:
        return {"_note": "empty"}
    n_dup = int(len(nonnull) - nonnull.nunique())
    lens = nonnull.str.len()
    return {"n_duplicates": n_dup,
            "uniqueness_ratio": float(nonnull.nunique() / len(nonnull)),
            "length_consistent": bool(lens.nunique() == 1),
            "length_min": int(lens.min()), "length_max": int(lens.max())}


def profile_entity(df: pd.DataFrame, types: dict, ecfg: dict, thresholds: dict) -> dict:
    top_k = thresholds["report"]["top_k_values"]
    heavy = thresholds.get("_heavy_sample", 200000)
    seed = thresholds.get("_seed", 42)
    out = {}
    for col, t in types.items():
        s = df[col]
        # sample for heavy family ops
        s_heavy = s
        if len(s) > heavy and t["family"] in ("categorical", "text", "pattern"):
            s_heavy = s.sample(heavy, random_state=seed)
        prof = {"_type": t["semantic_type"], "_family": t["family"]}
        prof.update(_universal(s, thresholds, top_k))
        fam, st = t["family"], t["semantic_type"]
        try:
            if fam == "numeric":
                prof.update(_profile_numeric(s, thresholds))
            elif st == "boolean_flag":
                prof.update(_profile_boolean(s, thresholds))
            elif st == "ordinal":
                order = (ecfg.get("column_overrides", {}).get(col, {}) or {}).get("order")
                prof.update(_profile_ordinal(s_heavy, order, thresholds))
            elif fam == "categorical":
                prof.update(_profile_categorical(s_heavy, thresholds))
            elif fam == "datetime":
                prof.update(_profile_datetime(s, ecfg, col, thresholds))
            elif fam == "pattern":
                prof.update(_profile_pattern(s_heavy, st, thresholds))
            elif fam == "text":
                prof.update(_profile_text(s_heavy, thresholds))
            elif fam == "keylike":
                prof.update(_profile_keylike(s, thresholds))
        except Exception as e:  # fail-soft: record, never crash the run
            prof["_profile_error"] = f"{type(e).__name__}: {e}"
        out[col] = prof
    return out



# ============================================================================
# Data-quality & sanity checks
# ============================================================================

def _column_quality(col: str, t: dict, prof: dict, thresholds: dict) -> dict:
    qa = thresholds["quality"]
    flags, actions = [], []
    status = "OK"
    pm = prof.get("pct_missing") or 0.0

    if pm >= qa["missing_drop_suggest"]:
        flags.append("very_high_missing")
        actions.append({"op": "drop", "reason": f"missing={pm:.2f}"})
        status = "DROP_SUGGESTED"
    elif pm >= qa["missing_warn"]:
        flags.append("high_missing")
        actions.append({"op": "impute", "strategy": "median_or_mode",
                        "add_missing_flag": True, "reason": f"missing={pm:.2f}"})
        status = "WARN"

    if prof.get("is_constant"):
        flags.append("constant")
        actions.append({"op": "drop", "reason": "single value"})
        status = "DROP_SUGGESTED"
    elif prof.get("is_near_constant"):
        flags.append("near_constant")
        if status == "OK":
            status = "WARN"

    if t["parse_fail_rate"] and t["parse_fail_rate"] > 0.05:
        flags.append("parse_failures")
        actions.append({"op": "cast", "to": t["semantic_type"],
                        "reason": f"parse_fail={t['parse_fail_rate']:.2f}"})
        if status == "OK":
            status = "WARN"

    # numeric range/sign sanity
    if t["family"] == "numeric":
        if t["semantic_type"] in ("amount", "count") and prof.get("n_negative", 0) > 0:
            flags.append("unexpected_negatives")
            actions.append({"op": "review_negatives", "n": prof["n_negative"]})
        out = prof.get("outliers", {})
        if out.get("fraction") and out["fraction"] > 0.02:
            actions.append({"op": "outlier_cap", "at": "p99",
                            "reason": f"outliers={out['fraction']:.3f}"})

    # datetime sanity
    if t["family"] == "datetime":
        if prof.get("n_future_dates", 0) > 0:
            flags.append("future_dates")
        if prof.get("n_sentinel_dates", 0) > 0:
            flags.append("sentinel_dates")

    # categorical hygiene
    if prof.get("case_whitespace_variants", 0) > 0:
        flags.append("case_whitespace_variants")
        actions.append({"op": "standardize_case_trim",
                        "reason": f"{prof['case_whitespace_variants']} collapsible variants"})

    # high-card non-key string -> likely mislabeled
    if t["feature_role"] == "feature" and t["family"] in ("categorical", "text") \
            and (prof.get("cardinality_ratio") or 0) >= qa["high_cardinality_ratio"]:
        flags.append("high_cardinality")

    # validity for pattern types
    if "validity_rate" in prof and prof["validity_rate"] is not None and prof["validity_rate"] < 0.9:
        flags.append("low_format_validity")
        actions.append({"op": "validate_format", "valid_rate": prof["validity_rate"]})

    return {"quality_status": status, "quality_flags": flags, "actions": actions}


def quality_entity(df: pd.DataFrame, types: dict, profiles: dict,
                   catalog_entry: dict, thresholds: dict) -> dict:
    qa = thresholds["quality"]
    col_quality = {col: _column_quality(col, types[col], profiles[col], thresholds)
                   for col in types}

    # entity-level
    pk = catalog_entry.get("primary_key") or []
    pk_unique = None
    if pk and all(c in df.columns for c in pk):
        pk_unique = bool(not df.duplicated(subset=pk).any())
    full_dup_rate = float(df.duplicated().mean()) if len(df) else 0.0

    statuses = [q["quality_status"] for q in col_quality.values()]
    n_drop = statuses.count("DROP_SUGGESTED")
    n_warn = statuses.count("WARN")
    n_cols = max(1, len(statuses))
    score = round(100 * (1 - (n_drop * 1.0 + n_warn * 0.4) / n_cols), 1)

    return {
        "columns": col_quality,
        "entity_checks": {
            "primary_key": pk, "primary_key_unique": pk_unique,
            "full_row_duplicate_rate": full_dup_rate,
            "schema_conflicts": catalog_entry.get("schema_conflicts", []),
            "n_columns": len(statuses), "n_drop_suggested": n_drop, "n_warn": n_warn,
        },
        "quality_score": score,
    }



# ============================================================================
# Within-entity multivariate / relationships
# ============================================================================

def _clean_matrix(dfm: pd.DataFrame) -> list:
    """DataFrame -> nested lists with NaN replaced by None (NaN is invalid JSON)."""
    return [[None if pd.isna(v) else round(float(v), 4) for v in row]
            for row in dfm.to_numpy()]


def _numeric_frame(df, types):
    cols = [c for c, t in types.items()
            if t["family"] == "numeric" and t["feature_role"] == "feature"]
    if not cols:
        return pd.DataFrame()
    return df[cols].apply(pd.to_numeric, errors="coerce")


def multivariate_entity(df: pd.DataFrame, types: dict, thresholds: dict) -> dict:
    corr_cfg = thresholds["correlation"]
    seed = thresholds.get("_seed", 42)
    heavy = thresholds.get("_heavy_sample", 200000)
    sample = df.sample(heavy, random_state=seed) if len(df) > heavy else df

    out = {"redundant_pairs": [], "top_numeric_corr": [], "top_spearman_corr": [],
           "nonlinear_pairs": [], "vif": [], "numeric_categorical_assoc": [],
           "categorical_associations": [], "directional_associations": [],
           "functional_dependencies": [], "co_missing_pairs": []}

    # numeric-numeric: Pearson (linear) + Spearman (monotonic, non-linear-robust)
    num = _numeric_frame(sample, types)
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] > corr_cfg["max_numeric_cols"]:
        num = num.iloc[:, :corr_cfg["max_numeric_cols"]]
    if num.shape[1] >= 2:
        pear = num.corr(method="pearson")
        spear = num.corr(method="spearman")
        cols = pear.columns
        pairs, spairs = [], []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, rs = pear.iloc[i, j], spear.iloc[i, j]
                if pd.notna(r):
                    pairs.append((cols[i], cols[j], float(r)))
                if pd.notna(rs):
                    spairs.append((cols[i], cols[j], float(rs)))
                # strong monotonic but weak linear -> non-linear relationship
                if pd.notna(r) and pd.notna(rs) and abs(rs) - abs(r) >= 0.2 and abs(rs) >= 0.5:
                    out["nonlinear_pairs"].append(
                        {"a": cols[i], "b": cols[j], "pearson": round(float(r), 4),
                         "spearman": round(float(rs), 4)})
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        spairs.sort(key=lambda x: abs(x[2]), reverse=True)
        out["top_numeric_corr"] = [{"a": a, "b": b, "pearson": round(r, 4)}
                                   for a, b, r in pairs[:corr_cfg["top_pairs"]]]
        out["top_spearman_corr"] = [{"a": a, "b": b, "spearman": round(r, 4)}
                                    for a, b, r in spairs[:corr_cfg["top_pairs"]]]
        out["redundant_pairs"] = [{"a": a, "b": b, "pearson": round(r, 4)}
                                  for a, b, r in pairs if abs(r) >= corr_cfg["redundant_abs"]]
        out["nonlinear_pairs"] = out["nonlinear_pairs"][:corr_cfg["top_pairs"]]
        # VIF: multi-way collinearity (catches what pairwise corr can't)
        vif = metrics.variance_inflation_factors(num.iloc[:, :30])
        if vif:
            out["vif"] = [{"column": c, "vif": round(v, 3)}
                          for c, v in sorted(vif.items(), key=lambda kv: -kv[1])]
        # full Pearson + Spearman matrices (for the heatmap), capped for size/legibility
        hm = corr_cfg.get("heatmap_max_cols", 40)
        mcols = list(cols)[:hm]
        out["corr_matrix"] = {
            "columns": mcols,
            "pearson": _clean_matrix(pear.loc[mcols, mcols]),
            "spearman": _clean_matrix(spear.loc[mcols, mcols]),
        }

    # categorical-categorical: Cramer's V (symmetric strength) + Theil's U (direction)
    cat_cols = [c for c, t in types.items()
                if t["family"] == "categorical" and t["feature_role"] == "feature"
                and df[c].nunique() <= 50][:30]
    cv_cache = {}
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            a, b = cat_cols[i], cat_cols[j]
            sa, sb = sample[a].astype("object"), sample[b].astype("object")
            v = metrics.cramers_v(sa, sb)
            cv_cache[(i, j)] = v
            if v is not None and v >= 0.5:
                out["categorical_associations"].append(
                    {"a": a, "b": b, "cramers_v": round(v, 4)})
                # asymmetric: how much each side explains the other
                u_ab = metrics.theils_u(sa, sb)   # U(a|b)
                u_ba = metrics.theils_u(sb, sa)    # U(b|a)
                if u_ab is not None and u_ba is not None:
                    out["directional_associations"].append(
                        {"a": a, "b": b, "u_a_given_b": round(u_ab, 4),
                         "u_b_given_a": round(u_ba, 4)})
    # Cramer's V matrix (for the categorical heatmap), capped for legibility
    M = min(len(cat_cols), corr_cfg.get("heatmap_max_cols", 40))
    if M >= 2:
        mat = [[1.0 if i == j else
                (lambda v: None if v is None else round(v, 4))(cv_cache.get((min(i, j), max(i, j))))
                for j in range(M)] for i in range(M)]
        out["cramers_matrix"] = {"columns": cat_cols[:M], "values": mat}
    out["categorical_associations"].sort(key=lambda d: d["cramers_v"], reverse=True)
    out["categorical_associations"] = out["categorical_associations"][:corr_cfg["top_pairs"]]
    out["directional_associations"] = out["directional_associations"][:corr_cfg["top_pairs"]]

    # numeric <-> categorical: correlation ratio (eta), the analogue of |corr|
    num_cols = [c for c, t in types.items()
                if t["family"] == "numeric" and t["feature_role"] == "feature"][:30]
    eta_pairs = []
    for cc in cat_cols:
        for nc in num_cols:
            eta = metrics.correlation_ratio(sample[cc].astype("object"), sample[nc])
            if eta is not None and eta >= 0.3:
                eta_pairs.append({"categorical": cc, "numeric": nc, "eta": round(eta, 4)})
    eta_pairs.sort(key=lambda d: d["eta"], reverse=True)
    out["numeric_categorical_assoc"] = eta_pairs[:corr_cfg["top_pairs"]]

    # functional dependencies among low-card / key columns (A -> B)
    fd_cands = [c for c, t in types.items()
                if t["family"] in ("categorical", "keylike")
                and df[c].nunique() <= 2000][:25]
    for a in fd_cands:
        ga = sample.groupby(a, observed=True)
        for b in fd_cands:
            if a == b:
                continue
            if (ga[b].nunique() <= 1).all():
                out["functional_dependencies"].append({"determinant": a, "depends": b})

    # missingness correlation: which columns tend to be null together (phi on the
    # missingness mask). High |phi| -> shared root cause / structural missingness.
    miss = sample.isna()
    part = miss.loc[:, (miss.sum() > 0) & (miss.sum() < len(miss))]
    if part.shape[1] >= 2:
        if part.shape[1] > corr_cfg["max_numeric_cols"]:
            part = part.iloc[:, :corr_cfg["max_numeric_cols"]]
        mc = part.astype(float).corr()
        cols = mc.columns
        mpairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = mc.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.5:
                    mpairs.append({"a": cols[i], "b": cols[j], "phi": round(float(r), 4)})
        mpairs.sort(key=lambda d: abs(d["phi"]), reverse=True)
        out["co_missing_pairs"] = mpairs[:corr_cfg["top_pairs"]]
    return out



# ============================================================================
# Temporal split suggestion
# ============================================================================

def drift_entity(shards: list[str], types: dict, catalog_entry: dict,
                 thresholds: dict) -> dict:
    # Shards here are arbitrary size-splits of one population, not separate batches,
    # so cross-shard distribution comparison is meaningless. We only derive a
    # leakage-safe train/valid split point from the entity's time column.
    seed = thresholds.get("_seed", 42)
    heavy = thresholds.get("_heavy_sample", 200000)
    out = {"suggested_time_split": None}

    tcol = catalog_entry.get("time_column")
    if tcol and tcol in types:
        try:
            df, _ = io.load_entity(shards, row_cap=heavy, per_shard_stratify=True,
                                   seed=seed, columns=[tcol])
            d = pd.to_datetime(df[tcol], errors="coerce").dropna()
            if not d.empty:
                out["suggested_time_split"] = d.quantile(0.8).isoformat()
        except Exception as e:
            out["_temporal_error"] = f"{type(e).__name__}: {e}"
    return out
