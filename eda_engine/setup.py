"""Stage 1 -- SETUP: load & validate config, then catalog the data: schemas,
row counts, schema drift across shards, and the per-entity load plan.
"""
from __future__ import annotations

import fnmatch
import os

import yaml

from .utils import io



# ============================================================================
# Config load, validation & run manifest
# ============================================================================

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str) -> dict:
    ds = _load_yaml(os.path.join(config_dir, "datasource.yaml"))
    th = _load_yaml(os.path.join(config_dir, "thresholds.yaml"))
    tg = _load_yaml(os.path.join(config_dir, "target.yaml"))
    if not ds:
        raise FileNotFoundError(f"datasource.yaml required in {config_dir}")
    th.setdefault("_seed", ds.get("sampling", {}).get("random_seed", 42))
    cfg = {"datasource": ds, "thresholds": th, "target": tg, "config_dir": config_dir}
    return cfg


def column_override(entity_cfg: dict, column: str) -> dict:
    """Exact name match wins over glob pattern (e.g. '*_FLAG')."""
    ov = entity_cfg.get("column_overrides", {}) or {}
    if column in ov:
        return ov[column] or {}
    for pat, spec in ov.items():
        if ("*" in pat or "?" in pat) and fnmatch.fnmatch(column, pat):
            return spec or {}
    return {}


def resolve_entities(cfg: dict) -> dict:
    """Discover shards per entity; skip missing optional entities."""
    ds = cfg["datasource"]
    root = ds["root"]
    resolved, skipped = {}, []
    for name, ecfg in ds["entities"].items():
        shards = io.resolve_shards(root, ecfg["glob"])
        if not shards:
            if ecfg.get("optional"):
                skipped.append({"entity": name, "reason": "no files (optional)"})
                continue
            skipped.append({"entity": name, "reason": "no files (required, MISSING)"})
            continue
        resolved[name] = {**ecfg, "shards": shards}
    return {"entities": resolved, "skipped": skipped}



# ============================================================================
# Catalog & discovery
# ============================================================================

def catalog_entity(name: str, ecfg: dict, sampling: dict) -> dict:
    shards_meta = [io.shard_metadata(s) for s in ecfg["shards"]]
    total_rows = sum(m["num_rows"] for m in shards_meta)
    total_bytes = sum(m["bytes"] for m in shards_meta)

    # Unified schema = union of columns across shards (banking dumps drift).
    all_cols, conflicts = [], []
    seen_types: dict[str, set] = {}
    col_presence: dict[str, int] = {}
    for m in shards_meta:
        for c in m["columns"]:
            if c not in all_cols:
                all_cols.append(c)
            col_presence[c] = col_presence.get(c, 0) + 1
            seen_types.setdefault(c, set()).add(m["arrow_types"][c])
    n_shards = len(shards_meta)
    for c in all_cols:
        if col_presence[c] != n_shards:
            conflicts.append({"column": c, "type": "missing_in_some_shards",
                              "present_in": col_presence[c], "of": n_shards})
        if len(seen_types[c]) > 1:
            conflicts.append({"column": c, "type": "dtype_mismatch",
                              "types": sorted(seen_types[c])})

    row_cap = sampling["profile_row_cap"]
    return {
        "entity": name, "grain": ecfg.get("grain"),
        "n_shards": n_shards, "total_rows": total_rows, "total_bytes": total_bytes,
        "unified_columns": all_cols, "n_columns": len(all_cols),
        "arrow_types": {c: sorted(seen_types[c])[0] for c in all_cols},
        "schema_conflicts": conflicts,
        "primary_key": ecfg.get("primary_key"),
        "foreign_keys": ecfg.get("foreign_keys", {}),
        "time_column": ecfg.get("time_column"),
        "load_plan": "sample" if total_rows > row_cap else "full",
        "shards": [{"path": m["path"], "rows": m["num_rows"]} for m in shards_meta],
    }


def build_catalog(resolved: dict, sampling: dict) -> dict:
    return {name: catalog_entity(name, ecfg, sampling)
            for name, ecfg in resolved["entities"].items()}
