#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp>=1.2.0",
#   "pandas>=2.0",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "pyarrow",
#   "pyyaml",
# ]
# ///
"""
eda_mcp_server.py  --  the "eda" MCP connector.

Exposes the data-agnostic EDA engine + its artifacts as MCP tools so an LLM
client (Claude Desktop, a company gateway, Juspay Grid, ...) can RUN, INSPECT
and EXTEND the EDA without writing code.

Design rule: the LLM never computes statistics itself. Read tools return the
profile/score artifacts the engine already wrote (fast, deterministic); compute
tools (`run_eda`, `evaluate_feature`) call into `eda_engine`. Tools are coarse
("a meaningful unit of work"), never one-per-statistic.

Run it:
    uv run eda_mcp_server.py                 # stdio  (Claude Desktop / local agents)
    uv run eda_mcp_server.py http            # http   -> http://127.0.0.1:8000/mcp
    uv run eda_mcp_server.py http 0.0.0.0 9000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Resolve paths relative to THIS file so the server works from any CWD, and make
# `eda_engine` importable (it lives next to this script).
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CONFIG = ROOT / "config"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mcp = FastMCP("eda")


# --------------------------------------------------------------------------- #
# helpers (read-only, stdlib json -- no heavy imports on this path)
# --------------------------------------------------------------------------- #
def _read(*parts: str) -> Any:
    """Load an artifact JSON, or return {'error': ...} if absent/unrun."""
    p = ART.joinpath(*parts)
    if not p.exists():
        return {"error": f"artifact not found: {p.relative_to(ROOT)} "
                         f"(run_eda first?)"}
    with open(p) as f:
        return json.load(f)


def _entity_dirs() -> list[str]:
    if not ART.exists():
        return []
    return sorted(d.name for d in ART.iterdir()
                  if d.is_dir() and (d / "profile.json").exists())


def _pick_target(tj: dict, target: Optional[str]) -> Optional[str]:
    results = tj.get("results", {})
    if target and target in results:
        return target
    prim = tj.get("primary_target")
    if prim in results and results[prim].get("supervised"):
        return prim
    return next((t for t, r in results.items() if r.get("supervised")), None)


_COLSIG = ("iv", "iv_band", "auc", "ks", "point_biserial", "cramers_v_target",
           "lift", "predictive_rank", "predictive_score", "leakage_suspect")


# --------------------------------------------------------------------------- #
# READ TOOLS
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_entities() -> list[dict]:
    """List every profiled entity with a one-line summary: rows, columns,
    shards, quality score, whether it has a supervised target, primary key and
    time column. Start here to see what data is available."""
    cat = _read("catalog.json")
    if isinstance(cat, dict) and cat.get("error"):
        return [cat]
    out = []
    for name in _entity_dirs():
        c = cat.get(name, {}) if isinstance(cat, dict) else {}
        prof = _read(name, "profile.json")
        tj = _read(name, "target.json")
        sup = isinstance(tj, dict) and any(
            r.get("supervised") for r in tj.get("results", {}).values())
        out.append({
            "entity": name,
            "rows": c.get("total_rows"),
            "columns": c.get("n_columns"),
            "shards": c.get("n_shards"),
            "quality_score": prof.get("quality_score") if isinstance(prof, dict) else None,
            "supervised": bool(sup),
            "active_targets": [t for t, r in (tj.get("results", {}) if isinstance(tj, dict) else {}).items()
                               if r.get("supervised")],
            "primary_key": c.get("primary_key"),
            "time_column": c.get("time_column"),
        })
    return out


@mcp.tool()
def get_catalog() -> dict:
    """Full data catalog: per-entity schema (unified columns), row counts,
    shard list, byte size, keys, time column, and any cross-shard schema drift."""
    return _read("catalog.json")


@mcp.tool()
def profile_entity(entity: str) -> dict:
    """Compact per-column profile for one entity: semantic type, family, role,
    %missing, uniqueness, plus the entity quality score and entity-level checks
    (PK uniqueness, duplicate rate). Use profile_column for a single column's
    full statistics."""
    prof = _read(entity, "profile.json")
    if prof.get("error"):
        return prof
    types, profs = prof.get("types", {}), prof.get("profiles", {})
    cols = []
    for c, p in profs.items():
        t = types.get(c, {})
        cols.append({
            "column": c,
            "semantic_type": t.get("semantic_type"),
            "family": t.get("family"),
            "feature_role": t.get("feature_role"),
            "pct_missing": p.get("pct_missing"),
            "n_unique": p.get("n_unique"),
            "cardinality_ratio": p.get("cardinality_ratio"),
        })
    return {"entity": entity, "quality_score": prof.get("quality_score"),
            "entity_checks": prof.get("entity_checks", {}),
            "n_columns": len(cols), "columns": cols}


@mcp.tool()
def profile_column(entity: str, column: str) -> dict:
    """Every statistic for ONE column (the whole profile dict at once): type +
    all univariate stats appropriate to it (numeric: min/max/mean/median/std/
    percentiles/skew/kurtosis/outliers/histogram; categorical: value counts,
    entropy, imbalance; datetime: span/gaps; etc.). This is the coarse-grained
    replacement for per-statistic tools."""
    prof = _read(entity, "profile.json")
    if prof.get("error"):
        return prof
    if column not in prof.get("profiles", {}):
        return {"error": f"column '{column}' not in entity '{entity}'",
                "available": list(prof.get("profiles", {}))[:50]}
    return {"entity": entity, "column": column,
            "type": prof.get("types", {}).get(column, {}),
            "profile": prof["profiles"][column]}


@mcp.tool()
def target_signal(entity: str, target: Optional[str] = None,
                  top: Optional[int] = None) -> dict:
    """Predictive power of each column against a label, ranked. Returns per
    column IV + band, AUC, KS, point-biserial, Cramer's V, lift, rank and a
    leakage flag. `target` defaults to the primary target; omit `top` for all."""
    tj = _read(entity, "target.json")
    if tj.get("error"):
        return tj
    tname = _pick_target(tj, target)
    if not tname:
        return {"entity": entity, "supervised": False,
                "reason": tj.get("reason", "no supervised target for this entity")}
    res = tj["results"][tname]
    cols = res.get("columns", {})
    rows = sorted(({"column": c, **{k: v.get(k) for k in _COLSIG if k in v}}
                   for c, v in cols.items()),
                  key=lambda r: (r.get("predictive_rank") is None,
                                 r.get("predictive_rank", 1e9)))
    if top:
        rows = rows[:top]
    return {"entity": entity, "target": tname,
            "positive_rate": res.get("positive_rate"),
            "n_pos": res.get("n_pos"), "n_neg": res.get("n_neg"),
            "n_scored": len(cols), "columns": rows}


@mcp.tool()
def relationships(entity: str, include_matrices: bool = False) -> dict:
    """Within-entity relationships: top Pearson/Spearman pairs, non-linear
    pairs, VIF (multicollinearity), Cramer's V + Theil's U (categorical),
    numeric-vs-categorical eta, functional dependencies, redundant pairs and
    co-missing columns. Big NxN correlation/Cramer matrices are omitted unless
    include_matrices=True (they are large)."""
    prof = _read(entity, "profile.json")
    if prof.get("error"):
        return prof
    rel = dict(prof.get("relationships", {}))
    if not include_matrices:
        rel.pop("corr_matrix", None)
        rel.pop("cramers_matrix", None)
    return {"entity": entity, "relationships": rel}


@mcp.tool()
def entity_overlap() -> dict:
    """Cross-entity key overlap in the hub id-space (e.g. ACCOUNT_NUMBER): for
    each pair of entities how many key values are shared, the count common to
    ALL entities, and the union -- entities lacking the key are bridged through
    the hub via their foreign key."""
    r = _read("relational.json")
    if isinstance(r, dict) and r.get("error"):
        return r
    return {"key_overlap": r.get("key_overlap", []),
            "relationships": r.get("relationships", [])}


@mcp.tool()
def feature_plan(entity: Optional[str] = None) -> dict:
    """The feature-engineering contract. With `entity`: per-column recommended
    ops, role, priority, quality status, target signal, leakage + requires_human
    flags. Without it: the global plan incl. cross-entity aggregate suggestions."""
    if entity:
        fp = _read(entity, "feature_plan.json")
        if isinstance(fp, dict) and fp.get("error"):
            return fp
        return {"entity": entity, "plan": list(fp.values()) if isinstance(fp, dict) else fp}
    return _read("feature_plan.json")


@mcp.tool()
def quality_report(entity: Optional[str] = None) -> dict:
    """Data-quality roll-up: per entity the quality score, PK uniqueness,
    duplicate rate, and the columns flagged DROP_SUGGESTED / WARN. Omit `entity`
    for all entities."""
    def one(name: str) -> dict:
        prof = _read(name, "profile.json")
        fp = _read(name, "feature_plan.json")
        drops, warns = [], []
        if isinstance(fp, dict) and not fp.get("error"):
            for col, r in fp.items():
                st = r.get("quality_status")
                if st == "DROP_SUGGESTED":
                    drops.append(col)
                elif st == "WARN":
                    warns.append(col)
        return {"entity": name,
                "quality_score": prof.get("quality_score") if isinstance(prof, dict) else None,
                "entity_checks": prof.get("entity_checks", {}) if isinstance(prof, dict) else {},
                "drop_suggested": drops, "warn": warns}
    if entity:
        return one(entity)
    return {"entities": [one(n) for n in _entity_dirs()]}


@mcp.tool()
def search_columns(min_iv: Optional[float] = None,
                   semantic_type: Optional[str] = None,
                   feature_role: Optional[str] = None,
                   max_missing: Optional[float] = None,
                   requires_human: Optional[bool] = None,
                   entity: Optional[str] = None,
                   target: Optional[str] = None,
                   limit: int = 50) -> dict:
    """Find columns matching criteria across entities -- the LLM's way to locate
    raw material for features. Filters: min_iv, semantic_type, feature_role,
    max_missing (<=), requires_human, entity. Results are ranked by IV desc."""
    ents = [entity] if entity else _entity_dirs()
    hits = []
    for name in ents:
        prof = _read(name, "profile.json")
        if not isinstance(prof, dict) or prof.get("error"):
            continue
        types, profs = prof.get("types", {}), prof.get("profiles", {})
        tj = _read(name, "target.json")
        tname = _pick_target(tj, target) if isinstance(tj, dict) else None
        tcols = tj["results"][tname]["columns"] if tname else {}
        fp = _read(name, "feature_plan.json")
        for c, t in types.items():
            p = profs.get(c, {})
            iv = (tcols.get(c) or {}).get("iv")
            miss = p.get("pct_missing")
            rh = (fp.get(c) or {}).get("requires_human") if isinstance(fp, dict) else None
            if semantic_type and t.get("semantic_type") != semantic_type:
                continue
            if feature_role and t.get("feature_role") != feature_role:
                continue
            if min_iv is not None and (iv is None or iv < min_iv):
                continue
            if max_missing is not None and (miss is not None and miss > max_missing):
                continue
            if requires_human is not None and bool(rh) != requires_human:
                continue
            hits.append({"entity": name, "column": c,
                         "semantic_type": t.get("semantic_type"),
                         "feature_role": t.get("feature_role"),
                         "pct_missing": miss, "iv": iv,
                         "target": tname, "requires_human": rh})
    hits.sort(key=lambda h: (h["iv"] is None, -(h["iv"] or 0)))
    return {"n_matches": len(hits), "target_used": target, "columns": hits[:limit]}


# --------------------------------------------------------------------------- #
# COMPUTE TOOLS  (lazy-import eda_engine only when actually called)
# --------------------------------------------------------------------------- #
@mcp.tool()
def run_eda(entities: Optional[list[str]] = None, report: bool = False) -> dict:
    """(Re)run the EDA engine and refresh all artifacts. `entities` limits to a
    subset (default: all). Heavy -- profiles full data. Returns per-entity
    quality scores, supervised flag and elapsed time."""
    from eda_engine import orchestrator
    return orchestrator.run(str(CONFIG), only_entities=entities, make_report=report)


@mcp.tool()
def evaluate_feature(spec: dict) -> dict:
    """Compute a PROPOSED feature on the real data and score it against a target
    -- the propose->compute->score loop. The LLM proposes a spec; this returns
    the feature's IV/band, AUC, KS, point-biserial, lift and a leakage flag, so
    only features that actually move the target are kept. The LLM never computes
    the numbers itself.

    Single-entity spec shapes (numeric output):
      {"op":"transform","entity":E,"column":C,"fn":"log1p|sqrt|zscore|abs|reciprocal"}
      {"op":"ratio","entity":E,"numerator":C1,"denominator":C2}
      {"op":"interaction","entity":E,"a":C1,"b":C2}              # product
    Optional "target": <name>  (defaults to the entity's primary target).
    (Cross-entity aggregate features are a planned follow-up.)
    """
    import numpy as np
    import pandas as pd
    from eda_engine import setup, score
    from eda_engine.utils import io, metrics

    op = spec.get("op")
    entity = spec.get("entity")
    if not op or not entity:
        return {"error": "spec needs at least {'op', 'entity'}"}

    cfg = setup.load_config(str(CONFIG))
    resolved = setup.resolve_entities(cfg)
    if entity not in resolved["entities"]:
        return {"error": f"unknown entity '{entity}'",
                "available": list(resolved["entities"])}
    sampling = cfg["datasource"]["sampling"]

    # which columns do we need to load?
    if op == "transform":
        src = [spec["column"]]
    elif op == "ratio":
        src = [spec["numerator"], spec["denominator"]]
    elif op == "interaction":
        src = [spec["a"], spec["b"]]
    else:
        return {"error": f"unsupported op '{op}' "
                         "(transform|ratio|interaction)"}

    # resolve the target + the entity key needed to attach the label
    tctx_map = score.resolve_targets(cfg, resolved)
    tname = spec.get("target") or score.primary_target_name(cfg)
    tctx = (tctx_map or {}).get(tname)
    if not tctx or not tctx.get("available"):
        return {"error": f"target '{tname}' unavailable",
                "available_targets": [t for t, c in (tctx_map or {}).items()
                                      if c.get("available")]}
    applies = tctx["spec"].get("applies_to", {}).get(entity)
    if not applies:
        return {"error": f"target '{tname}' is not applicable to entity '{entity}'"}
    key = applies["key"]

    need = sorted({*src, key})
    df, _ = io.load_entity(resolved["entities"][entity]["shards"],
                           row_cap=sampling["heavy_op_sample"],
                           per_shard_stratify=sampling["per_shard_stratify"],
                           seed=sampling["random_seed"], columns=need)
    missing = [c for c in src if c not in df.columns]
    if missing:
        return {"error": f"columns not found in {entity}: {missing}"}

    y = score.label_series(entity, df, tctx, resolved)
    if y is None:
        return {"error": f"could not attach target '{tname}' to '{entity}'"}

    def num(col):
        # robust: drop currency symbols / commas / spaces, keep digits, sign,
        # decimal and exponent -- so amount-strings ("₹50,000") parse too.
        cleaned = df[col].astype(str).str.replace(r"[^0-9.+\-eE]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    # build the feature
    if op == "transform":
        x = num(spec["column"]); fn = spec.get("fn", "log1p")
        if fn == "log1p":
            x = np.log1p(x.clip(lower=0))
        elif fn == "sqrt":
            x = np.sqrt(x.clip(lower=0))
        elif fn == "zscore":
            sd = x.std(); x = (x - x.mean()) / sd if sd else x * 0
        elif fn == "abs":
            x = x.abs()
        elif fn == "reciprocal":
            x = 1.0 / x.replace(0, np.nan)
        else:
            return {"error": f"unknown fn '{fn}'"}
        name = spec.get("name", f"{spec['column']}__{fn}")
    elif op == "ratio":
        d = num(spec["denominator"]).replace(0, np.nan)
        x = num(spec["numerator"]) / d
        name = spec.get("name", f"{spec['numerator']}__over__{spec['denominator']}")
    else:  # interaction
        x = num(spec["a"]) * num(spec["b"])
        name = spec.get("name", f"{spec['a']}__x__{spec['b']}")

    x = x.replace([np.inf, -np.inf], np.nan)
    tc = cfg["thresholds"]["target"]
    iv, _ = metrics.iv_woe_numeric(x, y, tc["max_woe_bins"], tc["woe_min_bin_frac"])
    auc = metrics.single_feature_auc(x, y)
    band = metrics.iv_band(iv, tc["iv_weak"], tc["iv_medium"], tc["iv_strong"])
    leak = bool((auc is not None and auc >= tc["leakage_auc"]) or (iv or 0) > 5)

    # baseline: best IV among the source columns (so the LLM sees if it helped)
    tj = _read(entity, "target.json")
    base = {}
    if isinstance(tj, dict) and tname in tj.get("results", {}):
        bc = tj["results"][tname]["columns"]
        base = {c: (bc.get(c) or {}).get("iv") for c in src}

    return {"name": name, "spec": spec, "target": tname,
            "n": int(x.notna().sum()), "positive_rate": float(y.mean()),
            "iv": iv, "iv_band": band, "auc": auc,
            "ks": metrics.ks_statistic(x, y),
            "point_biserial": metrics.point_biserial(x, y),
            "lift": metrics.decile_lift(x, y, tc["max_woe_bins"]),
            "leakage_suspect": leak,
            "baseline_source_iv": base,
            "verdict": ("leakage_suspect" if leak else
                        "useful" if (iv or 0) >= tc["iv_medium"] else
                        "weak")}


# --------------------------------------------------------------------------- #
# transports
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "http":
        mcp.settings.host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
        mcp.settings.port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio
