"""Semantic-type catalog + inference rules. 100% data-agnostic.

`semantic_type` is what the whole pipeline branches on. Physical dtype is not
enough (PINCODE is int but categorical; ANNUAL_INCOME is a number stored as a
string; *_FLAG strings are booleans). Config overrides always win; otherwise we
infer from a sample using the ladder in `infer_semantic_type`.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# Pattern types: validity is checked later via these regexes.
RE_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
RE_IPV6 = re.compile(r"^[0-9a-fA-F:]+:[0-9a-fA-F:]+$")
RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RE_VPA = re.compile(r"^[\w.\-]+@[\w.\-]+$")          # UPI handle user@psp
RE_IFSC = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
RE_HEXHASH = re.compile(r"^[0-9a-fA-F]{16,}$")
RE_NUMERICISH = re.compile(r"^[\s]*[-+]?[\d,]*\.?\d+([eE][-+]?\d+)?[\s]*$")

# Families used by downstream stages to pick which analyses apply.
NUMERIC_TYPES = {"continuous", "amount", "count", "amount_string"}
CATEGORICAL_TYPES = {"categorical", "ordinal", "boolean_flag", "binary", "categorical_id"}
DATETIME_TYPES = {"datetime"}
PATTERN_TYPES = {"ip_address", "email", "vpa", "bank_code", "phone", "pincode"}
KEYLIKE_TYPES = {"identifier", "hashed_pii"}
TEXT_TYPES = {"free_text"}

# Core coded booleans plus common worded pairs (active/inactive, open/closed,
# success/fail, ...). Gender-style pairs (m/f, male/female) are deliberately
# excluded: they are genuine binary *categories*, not truth values, and map
# better as `binary`. Declare them in config if a boolean reading is wanted.
BOOL_TRUE = {"y", "yes", "true", "t", "1", "1.0",
             "active", "enabled", "on", "present", "success", "pass", "passed",
             "approved", "accept", "accepted", "valid", "open", "positive",
             "available", "complete", "completed"}
BOOL_FALSE = {"n", "no", "false", "f", "0", "0.0",
              "inactive", "disabled", "off", "absent", "failure", "failed",
              "fail", "rejected", "reject", "declined", "decline", "invalid",
              "closed", "negative", "unavailable", "incomplete"}
BOOL_TOKENS = BOOL_TRUE | BOOL_FALSE


def _frac_match(sample: pd.Series, predicate) -> float:
    if sample.empty:
        return 0.0
    return float(sample.map(predicate).mean())


def _try_numeric_frac(sample: pd.Series) -> float:
    parsed = pd.to_numeric(
        sample.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce")
    return float(parsed.notna().mean())


def _try_datetime_frac(sample: pd.Series) -> float:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def infer_semantic_type(series: pd.Series, thresholds: dict, *,
                        is_key: bool = False) -> dict:
    """Return {semantic_type, confidence, evidence}. Never raises."""
    ti = thresholds["type_inference"]
    qa = thresholds["quality"]
    n = len(series)
    nonnull = series.dropna()
    if nonnull.empty:
        return {"semantic_type": "constant", "confidence": 1.0,
                "evidence": "all-null"}

    samp = nonnull
    if len(samp) > ti["sample_for_inference"]:
        samp = samp.sample(ti["sample_for_inference"],
                           random_state=thresholds.get("_seed", 42))

    n_unique = nonnull.nunique()
    uniq_ratio = n_unique / len(nonnull)
    phys_numeric = pd.api.types.is_numeric_dtype(series)
    phys_dt = pd.api.types.is_datetime64_any_dtype(series)
    str_samp = samp.astype(str).str.strip()
    lower = str_samp.str.lower()

    # 1) identifier (uniqueness) -- only if declared key, else may be free text
    if is_key and uniq_ratio >= qa["id_uniqueness_min"]:
        return {"semantic_type": "identifier", "confidence": 0.95,
                "evidence": f"uniq_ratio={uniq_ratio:.3f}, key"}

    # 2) constant
    if n_unique <= 1:
        return {"semantic_type": "constant", "confidence": 1.0, "evidence": "1 value"}

    # 3) boolean flag -- coded (y/n, 0/1) or worded pairs (active/inactive, ...).
    #    Require the values to span BOTH a true- and a false-side token so a set
    #    of same-polarity synonyms isn't misread as a degenerate boolean.
    if n_unique <= ti["boolean_max_unique"] and lower.isin(BOOL_TOKENS).mean() > 0.95:
        distinct = set(lower.unique())
        if (distinct & BOOL_TRUE) and (distinct & BOOL_FALSE):
            return {"semantic_type": "boolean_flag", "confidence": 0.9,
                    "evidence": f"{n_unique} bool-coded values"}

    # 4) datetime
    if phys_dt:
        return {"semantic_type": "datetime", "confidence": 0.99, "evidence": "dt dtype"}
    if not phys_numeric and _try_datetime_frac(samp) >= ti["datetime_parse_success_min"]:
        return {"semantic_type": "datetime", "confidence": 0.85, "evidence": "parses as date"}

    # 5) pattern types (string)
    if not phys_numeric:
        if _frac_match(str_samp, lambda s: bool(RE_IFSC.match(s))) > 0.9:
            return {"semantic_type": "bank_code", "confidence": 0.9, "evidence": "IFSC pattern"}
        if _frac_match(str_samp, lambda s: bool(RE_EMAIL.match(s))) > 0.9:
            return {"semantic_type": "email", "confidence": 0.9, "evidence": "email pattern"}
        if _frac_match(str_samp, lambda s: bool(RE_IPV4.match(s) or RE_IPV6.match(s))) > 0.9:
            return {"semantic_type": "ip_address", "confidence": 0.9, "evidence": "ip pattern"}
        if _frac_match(str_samp, lambda s: bool(RE_VPA.match(s))) > 0.9:
            return {"semantic_type": "vpa", "confidence": 0.85, "evidence": "vpa pattern"}
        if _frac_match(str_samp, lambda s: bool(RE_HEXHASH.match(s))) > 0.9 and uniq_ratio > 0.5:
            return {"semantic_type": "hashed_pii", "confidence": 0.8, "evidence": "hex hash"}

    # 6) numbers-stored-as-string
    if not phys_numeric:
        num_frac = _try_numeric_frac(samp)
        if num_frac >= ti["numeric_parse_success_min"]:
            return {"semantic_type": "amount_string", "confidence": 0.8,
                    "evidence": f"numeric_parse={num_frac:.2f}"}

    # 7) numeric branch
    if phys_numeric:
        s = nonnull
        is_int = np.array_equal(s.dropna(), s.dropna().round()) if len(s) else False
        if is_int and (s.min() >= 0) and n_unique <= ti["categorical_max_unique"]:
            return {"semantic_type": "categorical_id", "confidence": 0.6,
                    "evidence": f"int, {n_unique} levels"}
        if is_int and (s.min() >= 0):
            return {"semantic_type": "count", "confidence": 0.6,
                    "evidence": "non-negative integers"}
        return {"semantic_type": "continuous", "confidence": 0.7, "evidence": "float"}

    # 8) categorical vs free text (strings)
    if n_unique <= ti["categorical_max_unique"] or uniq_ratio <= ti["categorical_max_ratio"]:
        st = "binary" if n_unique == 2 else "categorical"
        return {"semantic_type": st, "confidence": 0.7,
                "evidence": f"{n_unique} levels, ratio={uniq_ratio:.3f}"}
    if uniq_ratio >= qa["high_cardinality_ratio"]:
        avg_len = float(str_samp.str.len().mean())
        if avg_len > 20 or str_samp.str.contains(r"\s").mean() > 0.5:
            return {"semantic_type": "free_text", "confidence": 0.7,
                    "evidence": f"high-card, avg_len={avg_len:.1f}"}
        return {"semantic_type": "identifier", "confidence": 0.6,
                "evidence": f"high-card uniq_ratio={uniq_ratio:.3f}"}

    return {"semantic_type": "unknown", "confidence": 0.3, "evidence": "no rule matched"}


def family(semantic_type: str) -> str:
    if semantic_type in NUMERIC_TYPES:
        return "numeric"
    if semantic_type in CATEGORICAL_TYPES:
        return "categorical"
    if semantic_type in DATETIME_TYPES:
        return "datetime"
    if semantic_type in PATTERN_TYPES:
        return "pattern"
    if semantic_type in KEYLIKE_TYPES:
        return "keylike"
    if semantic_type in TEXT_TYPES:
        return "text"
    return "other"
