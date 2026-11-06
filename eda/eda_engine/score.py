"""Stage 3 -- SCORE: rank columns against the configured label(s) with
target-aware IV/WoE/AUC/KS/MI + a leakage guard, and characterize cross-entity
joins + the aggregation menu the FE step builds aggregate features from.
Auto-skips (no error) when no label is configured/available.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .utils import metrics, io



# ============================================================================
# Target-aware analysis
# ============================================================================

def active_target_names(cfg: dict) -> list[str]:
    """Targets to analyze; supports `active_targets` (list) or `active_target`."""
    tgt_cfg = cfg.get("target") or {}
    targets = tgt_cfg.get("targets") or {}
    names = tgt_cfg.get("active_targets")
    if not names:
        single = tgt_cfg.get("active_target")
        names = [single] if single else []
    return [n for n in names if n in targets]


def primary_target_name(cfg: dict) -> str | None:
    tgt_cfg = cfg.get("target") or {}
    names = active_target_names(cfg)
    return tgt_cfg.get("primary_target") or (names[0] if names else None)


def resolve_targets(cfg: dict, resolved: dict) -> dict:
    """Resolve every active target -> {name: tctx}. Unavailable ones included
    with available=False so the run can degrade per-target without error."""
    out = {}
    for name in active_target_names(cfg):
        out[name] = resolve_target(cfg, resolved, name)
    return out


def resolve_target(cfg: dict, resolved: dict, name: str | None = None) -> dict:
    """Return target context for one target, or available=False."""
    tgt_cfg = cfg.get("target") or {}
    targets = tgt_cfg.get("targets") or {}
    active = name or primary_target_name(cfg)
    if not active or active not in targets:
        return {"available": False, "reason": "no active target configured"}
    spec = targets[active]
    entities = resolved["entities"]
    seed = cfg["thresholds"].get("_seed", 42)

    # Build positive value sets keyed by the LEFT column name.
    positive_keys: dict[str, set] = {}
    used_sources = []
    for src in spec.get("positive_if_present_in", []):
        ent = src["entity"]
        if ent not in entities:
            continue  # label entity missing -> just skip this source
        left = src["join"]["left"]
        right = src["join"]["right"]
        # Label entities are small; read all columns so an optional filter works.
        df = _read_all(entities[ent]["shards"])
        if df is None or right not in df.columns:
            continue
        if src.get("filter"):
            try:
                df = df.query(src["filter"])
            except Exception:
                pass  # bad filter -> ignore, use unfiltered set
        vals = set(df[right].dropna().astype(str).unique())
        positive_keys.setdefault(left, set()).update(vals)
        used_sources.append({"entity": ent, "n_positive_keys": len(vals)})

    if not positive_keys:
        return {"available": False, "reason": "no label rows found"}
    return {"available": True, "name": active, "spec": spec,
            "positive_keys": positive_keys, "sources": used_sources,
            "default_negative": spec.get("default_negative", True), "_seed": seed}


def _read_all(shards):
    frames = [pd.read_parquet(s) for s in shards]
    return pd.concat(frames, ignore_index=True) if frames else None


def _read_cols(shards, cols):
    """Read only `cols` that exist in each shard, concatenated."""
    import pyarrow.parquet as pq
    frames = []
    for s in shards:
        names = set(pq.read_schema(s).names)
        use = [c for c in cols if c in names]
        if not use:
            continue
        frames.append(pd.read_parquet(s, columns=use))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def label_series(entity: str, df: pd.DataFrame, tctx: dict, resolved: dict) -> pd.Series | None:
    """Per-row 0/1 label for a feature entity, or None if not applicable."""
    spec = tctx["spec"]
    applies = spec.get("applies_to", {})
    if entity not in applies:
        return None
    rule = applies[entity]
    key = rule["key"]
    via = rule.get("via")
    pos = tctx["positive_keys"]

    if not via:
        # entity's key directly matches a positive-key column
        if key not in pos or key not in df.columns:
            return None
        y = df[key].astype(str).isin(pos[key]).astype(int)
        return y

    # bridge through another entity (e.g. customers -> accounts)
    if via not in resolved["entities"]:
        return None
    # which positive-key column does the bridge carry?
    bridge_shards = resolved["entities"][via]["shards"]
    import pyarrow.parquet as pq
    bridge_names = set(pq.read_schema(bridge_shards[0]).names)
    left_key = next((k for k in pos if k in bridge_names), None)
    if left_key is None or key not in bridge_names or key not in df.columns:
        return None
    bridge = _read_cols(bridge_shards, [key, left_key])
    if bridge is None:
        return None
    pos_bridge = bridge[bridge[left_key].astype(str).isin(pos[left_key])][key].astype(str).unique()
    y = df[key].astype(str).isin(set(pos_bridge)).astype(int)
    return y


def target_entity(df: pd.DataFrame, types: dict, y: pd.Series, thresholds: dict) -> dict:
    tc = thresholds["target"]
    seed = thresholds.get("_seed", 42)
    heavy = thresholds.get("_heavy_sample", 200000)
    if len(df) > heavy:
        idx = df.sample(heavy, random_state=seed).index
        df, y = df.loc[idx], y.loc[idx]

    pos_rate = float(y.mean())
    out = {"supervised": True, "positive_rate": pos_rate,
           "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "columns": {}}
    if y.nunique() < 2:
        out["supervised"] = False
        out["reason"] = "label has single class after join"
        return out

    ranked = []
    for col, t in types.items():
        if t["feature_role"] != "feature":
            continue
        rec = {}
        fam = t["family"]
        try:
            if fam == "numeric":
                x = pd.to_numeric(df[col], errors="coerce")
                # numeric IV is already binned to <= max_woe_bins; we keep only the
                # scalar score, not the bin table (FE recomputes WoE on train data).
                iv, _bins = metrics.iv_woe_numeric(x, y, tc["max_woe_bins"], tc["woe_min_bin_frac"])
                rec = {"iv": iv, "iv_band": metrics.iv_band(iv, tc["iv_weak"], tc["iv_medium"], tc["iv_strong"]),
                       "auc": metrics.single_feature_auc(x, y),
                       "ks": metrics.ks_statistic(x, y),
                       "point_biserial": metrics.point_biserial(x, y),
                       "lift": metrics.decile_lift(x, y, tc["max_woe_bins"])}
            elif fam in ("categorical", "pattern"):
                x = df[col].astype("object")
                # Cap high-cardinality columns to top-N + "__other__" so IV reflects
                # real signal, not per-value memorization (e.g. BRANCH_CODE, PINCODE,
                # VPA, IP). Keeps the artifact tiny; the bin table is never persisted.
                n_distinct = int(x.nunique())
                cap = tc["max_woe_bins"]
                capped = _cap_categorical(x, cap) if n_distinct > cap else x
                iv, _bins = metrics.iv_woe_categorical(capped, y)
                rec = {"iv": iv, "iv_band": metrics.iv_band(iv, tc["iv_weak"], tc["iv_medium"], tc["iv_strong"]),
                       "cramers_v_target": metrics.cramers_v(capped, y),
                       "lift": metrics.categorical_lift(capped, y),
                       "target_rate_by_cat": _target_rate(x, y)}
                if n_distinct > cap:
                    rec["high_cardinality_capped"] = {"n_distinct": n_distinct, "kept_top": cap}
            else:
                continue
        except Exception as e:
            rec = {"_error": f"{type(e).__name__}: {e}"}

        # leakage guard
        auc = rec.get("auc")
        if (auc is not None and auc >= tc["leakage_auc"]) or (rec.get("iv") or 0) > 5:
            rec["leakage_suspect"] = True
        out["columns"][col] = rec
        ranked.append((col, _score(rec)))

    ranked.sort(key=lambda x: (x[1] is not None, x[1] or 0), reverse=True)
    for rank, (col, sc) in enumerate(ranked, 1):
        out["columns"][col]["predictive_rank"] = rank
        out["columns"][col]["predictive_score"] = sc
    return out


def _score(rec: dict):
    parts = [rec.get("iv"), rec.get("auc") and (rec["auc"] - 0.5) * 2, abs(rec.get("point_biserial") or 0)]
    vals = [p for p in parts if p is not None]
    return float(max(vals)) if vals else None


def _cap_categorical(x: pd.Series, n: int) -> pd.Series:
    """Keep the n most frequent values; fold the rest into '__other__'. Prevents
    per-value WoE overfit on high-cardinality IDs/geo (BRANCH_CODE, PINCODE, VPA)."""
    xs = x.astype(str)
    top = set(xs.value_counts().head(n).index)
    return xs.where(xs.isin(top), "__other__")


def _target_rate(x: pd.Series, y: pd.Series, top=20):
    df = pd.DataFrame({"x": x.astype(str), "y": y})
    g = df.groupby("x", observed=True)["y"].agg(["mean", "count"]).sort_values("count", ascending=False).head(top)
    return {k: {"rate": float(v["mean"]), "n": int(v["count"])} for k, v in g.iterrows()}



# ============================================================================
# Cross-entity relational EDA
# ============================================================================

def relational_eda(resolved: dict, types_by_entity: dict, thresholds: dict) -> dict:
    seed = thresholds.get("_seed", 42)
    cap = thresholds.get("_heavy_sample", 200000)
    ents = resolved["entities"]
    rels = []

    for child, ecfg in ents.items():
        for fk, ref in (ecfg.get("foreign_keys") or {}).items():
            parent_ent, parent_col = ref["references"].split(".")
            if parent_ent not in ents:
                continue
            child_vals = _read_cols(ents[child]["shards"], [fk])
            parent_vals = _read_cols(ents[parent_ent]["shards"], [parent_col])
            if child_vals is None or parent_vals is None:
                continue
            cv = child_vals[fk].dropna().astype(str)
            pv = parent_vals[parent_col].dropna().astype(str)
            pset = set(pv.unique())
            orphan_rate = float((~cv.isin(pset)).mean()) if len(cv) else None
            fan = cv.value_counts()
            childless = float((~pv.isin(set(cv.unique()))).mean()) if len(pv) else None
            parent_unique = pv.is_unique
            card = "N:1" if parent_unique else "N:M"
            rels.append({
                "from": f"{child}.{fk}", "to": f"{parent_ent}.{parent_col}",
                "cardinality": card,
                "orphan_rate": orphan_rate,
                "parent_without_children_rate": childless,
                "fanout_stats": {"mean": float(fan.mean()), "median": float(fan.median()),
                                 "p95": float(fan.quantile(0.95)), "max": int(fan.max())}
                if len(fan) else None,
                "aggregation_candidates": _agg_menu(types_by_entity.get(child, {}), fk),
            })
    return {"relationships": rels,
            "key_overlap": key_overlap_analysis(resolved, thresholds)}


def key_overlap_analysis(resolved: dict, thresholds: dict) -> list:
    """Entity overlap measured in a common id-space, bridged through the hub entity.

    Every entity is projected onto each shared id ("space"). An entity that
    carries the id directly contributes its own distinct values; one that does
    not is *bridged* through the hub dimension table (the entity whose primary
    key is a shared id and which also carries another shared id), e.g. customers
    -> ACCOUNT_NUMBER via accounts, or mha_l* -> CUSTOMER_ID via accounts. This
    yields true cross-entity overlaps (pairwise + intersection across ALL
    entities) even where two entities share no column directly.

    100% data-agnostic: shared ids, the hub, and the bridge map are all derived
    from config keys + actual schemas; nothing about accounts/PNB is hard-coded.
    """
    import pyarrow.parquet as pq
    ents = resolved["entities"]

    # candidate id columns = any PK/FK name declared in config
    cand = set()
    for ecfg in ents.values():
        cand |= set(ecfg.get("primary_key") or [])
        cand |= set((ecfg.get("foreign_keys") or {}).keys())
    schema = {}
    for name, ecfg in ents.items():
        try:
            schema[name] = set(pq.read_schema(ecfg["shards"][0]).names)
        except Exception:
            schema[name] = set()
    # a "link key" is an id column physically present in >= 2 entities
    link_keys = [k for k in sorted(cand) if sum(k in s for s in schema.values()) >= 2]
    if not link_keys:
        return []

    # hub = a dimension entity whose primary key is a link key and which carries
    # >= 2 link keys, so it can translate one id-space into another.
    hub = None
    for name, ecfg in ents.items():
        pk = ecfg.get("primary_key") or []
        if pk and all(p in link_keys for p in pk) \
                and sum(k in schema[name] for k in link_keys) >= 2:
            hub = name
            break
    bridge_keys = [k for k in link_keys if hub and k in schema[hub]]
    bridge_df = None
    if hub and len(bridge_keys) >= 2:
        bridge_df = _read_cols(ents[hub]["shards"], bridge_keys)
        if bridge_df is not None:
            for k in bridge_keys:
                bridge_df[k] = bridge_df[k].astype(str)

    # distinct values of one column of one entity (memoized)
    own_cache = {}
    def own_set(name, key):
        if (name, key) not in own_cache:
            s = None
            if key in schema.get(name, set()):
                df = _read_cols(ents[name]["shards"], [key])
                if df is not None and key in df.columns:
                    s = set(df[key].dropna().astype(str).unique())
            own_cache[(name, key)] = s
        return own_cache[(name, key)]

    # One canonical overlap space = the hub's primary key (e.g. ACCOUNT_NUMBER).
    # Every entity is measured in this single id-space; entities lacking the
    # column are bridged through the hub via their foreign key. (Fall back to all
    # link keys only if no hub resolves.)
    hub_pk = None
    if hub:
        pk = ents[hub].get("primary_key") or []
        if len(pk) == 1 and pk[0] in link_keys:
            hub_pk = pk[0]
    spaces = [hub_pk] if hub_pk else link_keys

    out = []
    for K in spaces:
        proj, bridged = {}, {}
        for name in ents:
            if K in schema.get(name, set()):                 # direct
                s = own_set(name, K)
                if s is not None:
                    proj[name], bridged[name] = s, False
            elif bridge_df is not None and K in bridge_keys:  # bridge via hub
                for K2 in bridge_keys:
                    if K2 == K or K2 not in schema.get(name, set()):
                        continue
                    e2 = own_set(name, K2)
                    if not e2:
                        continue
                    s = set(bridge_df.loc[bridge_df[K2].isin(e2), K].dropna().unique())
                    if s:
                        proj[name], bridged[name] = s, True
                    break
        if len(proj) < 2:
            continue
        names = list(proj)
        counts = {n: len(proj[n]) for n in names}
        pairwise = [[(counts[a] if a == b else len(proj[a] & proj[b]))
                     for b in names] for a in names]
        common = set.intersection(*(proj[n] for n in names))
        union = set.union(*(proj[n] for n in names))
        out.append({
            "space": K, "hub": hub, "entities": names, "bridged": bridged,
            "distinct_counts": counts, "pairwise_overlap": pairwise,
            "common_all": len(common), "union_all": len(union),
        })
    return out


def _agg_menu(child_types: dict, fk: str) -> list:
    """Which child columns can roll up to the parent grain, and how."""
    menu = []
    for col, t in child_types.items():
        if col == fk or t["feature_role"] == "key":
            continue
        fam = t["family"]
        if fam == "numeric":
            ops = ["sum", "mean", "max", "min", "std"]
        elif fam == "categorical":
            ops = ["nunique", "mode", "entropy"]
        elif fam == "datetime":
            ops = ["min", "max", "recency_days", "span_days"]
        elif fam == "pattern":
            ops = ["nunique"]
        else:
            continue
        menu.append({"column": col, "semantic_type": t["semantic_type"], "ops": ops})
    # always offer a plain row count
    menu.insert(0, {"column": "*", "semantic_type": "count", "ops": ["count"]})
    return menu
