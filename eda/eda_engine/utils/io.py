"""IO: shard discovery, sampled loading, coercion, artifact persistence.

No domain knowledge -- everything is driven by paths/config passed in.
"""
from __future__ import annotations

import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def resolve_shards(root: str, pattern: str) -> list[str]:
    files = glob.glob(os.path.join(root, pattern))
    return sorted(files, key=lambda p: _natkey(os.path.basename(p)))


def shard_metadata(path: str) -> dict:
    md = pq.ParquetFile(path).metadata
    sch = pq.read_schema(path)
    return {"path": path, "num_rows": md.num_rows, "bytes": os.path.getsize(path),
            "columns": list(sch.names),
            "arrow_types": {n: str(t) for n, t in zip(sch.names, sch.types)}}


# --------------------------------------------------------------------------- #
# Loading (with stratified-across-shard sampling)
# --------------------------------------------------------------------------- #
def load_entity(shards: list[str], *, row_cap: int, per_shard_stratify: bool,
                seed: int, columns: list[str] | None = None):
    """Load shards into one DataFrame, capping rows. Returns (df, info)."""
    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    sampled = total > row_cap
    frames = []
    if not sampled:
        for s in shards:
            frames.append(pd.read_parquet(s, columns=columns))
    elif per_shard_stratify:
        per = max(1, row_cap // len(shards))
        for s in shards:
            df = pd.read_parquet(s, columns=columns)
            if len(df) > per:
                df = df.sample(per, random_state=seed)
            frames.append(df)
    else:
        for s in shards:
            frames.append(pd.read_parquet(s, columns=columns))
        df = pd.concat(frames, ignore_index=True)
        return df.sample(row_cap, random_state=seed).reset_index(drop=True), \
            {"total_rows": total, "sampled": True, "sample_rows": row_cap}
    df = pd.concat(frames, ignore_index=True)
    return df, {"total_rows": total, "sampled": sampled, "sample_rows": len(df)}


def load_shard_frames(shards: list[str], columns: list[str], *, per_cap: int,
                      seed: int) -> list[pd.DataFrame]:
    """One (optionally capped) frame per shard -- for cross-shard drift."""
    out = []
    for s in shards:
        df = pd.read_parquet(s, columns=[c for c in columns
                                         if c in pq.read_schema(s).names])
        if len(df) > per_cap:
            df = df.sample(per_cap, random_state=seed)
        out.append(df)
    return out


# --------------------------------------------------------------------------- #
# Coercion (shared by the typing plan and FE)
# --------------------------------------------------------------------------- #
def coerce(series: pd.Series, semantic_type: str):
    """Cast a column per its semantic type. Returns (coerced, parse_fail_rate)."""
    nonnull_before = series.notna().sum()
    out = series
    try:
        if semantic_type in ("amount_string", "continuous", "amount", "count"):
            if not pd.api.types.is_numeric_dtype(series):
                out = pd.to_numeric(
                    series.astype(str).str.replace(",", "", regex=False)
                    .str.replace("₹", "", regex=False).str.strip(),
                    errors="coerce")
        elif semantic_type == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = pd.to_datetime(series, errors="coerce")
        elif semantic_type == "boolean_flag":
            from .typing_registry import BOOL_TRUE
            out = series.astype(str).str.strip().str.lower().map(
                lambda v: 1 if v in BOOL_TRUE else (0 if v not in ("nan", "none", "") else np.nan))
    except Exception:
        return series, 0.0
    nonnull_after = out.notna().sum()
    fail = 0.0
    if nonnull_before > 0:
        fail = float(max(0, nonnull_before - nonnull_after) / nonnull_before)
    return out, fail


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if pd.isna(o) if np.isscalar(o) else False:
            return None
        return super().default(o)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


def write_parquet(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Object columns may mix dtypes (e.g. numeric min vs ISO-date min) -> stringify
    # so the flat table stays writable/queryable; pure-numeric cols are untouched.
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and v != v)
                              else str(v))
    df.to_parquet(path, index=False)


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
