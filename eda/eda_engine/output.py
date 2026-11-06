"""Stage 4 -- OUTPUT: synthesize the machine-readable feature plan from all
upstream metrics, and render the flat columns table + the markdown report --
both built from the same in-memory results so they never disagree.
"""
from __future__ import annotations

import os

import pandas as pd



# ============================================================================
# Feature-plan synthesis
# ============================================================================

def _recommend_ops(col, t, prof, q, tgt, thresholds):
    st, fam = t["semantic_type"], t["family"]
    ops, requires_human, notes = [], False, []

    # quality-driven ops first (cast/impute/clean), reuse quality actions verbatim
    for a in q.get("actions", []):
        if a["op"] in ("drop", "outlier_cap", "standardize_case_trim", "cast", "impute"):
            ops.append(a)

    if st == "boolean_flag":
        ops.append({"op": "map_binary", "to": [0, 1]})
    elif st == "ordinal":
        ops.append({"op": "ordinal_encode", "by": "config_order"})
    elif fam == "categorical":
        ratio = prof.get("cardinality_ratio") or 0
        if ratio >= thresholds["quality"]["high_cardinality_ratio"]:
            ops.append({"op": "encode", "method": "target_or_frequency_or_hash"})
            requires_human = True
            notes.append("high-cardinality grouping may need domain knowledge")
        else:
            method = "woe" if (tgt and tgt.get("iv")) else "onehot"
            ops.append({"op": "encode", "method": method})
        if prof.get("n_rare_levels", 0) > 0:
            ops.append({"op": "rare_bucket", "min_freq_frac": 0.01})
    elif fam == "numeric":
        if prof.get("log_transform_candidate"):
            ops.append({"op": "transform", "fn": "log1p",
                        "reason": f"skew={prof.get('skew')}, all_positive"})
        if prof.get("zero_inflation", 0) > 0.3:
            ops.append({"op": "add_flag", "name": f"{col}__is_zero"})
        ops.append({"op": "scale", "method": "robust"})
    elif fam == "datetime":
        ops.append({"op": "datetime_decompose", "parts": prof.get("derived_feature_stubs", [])})
    elif fam == "pattern":
        stubs = prof.get("decomposition_stubs", [])
        for s in stubs:
            if "requires_human" in s:
                requires_human = True
                notes.append(f"{s}")
            else:
                ops.append({"op": "extract", "feature": s})
    elif fam == "text":
        ops.append({"op": "text_basic", "features": ["char_len", "token_count", "has_digit"]})
        ops.append({"op": "vectorize", "status": "deferred"})
        requires_human = True
        notes.append("text vectorization deferred to later/LLM stage")
    elif fam == "keylike":
        return {"feature_role": "key", "recommended_ops": [],
                "requires_human": False, "notes": ["used for joins/aggregation only"]}

    return {"feature_role": "feature", "recommended_ops": ops,
            "requires_human": requires_human, "notes": notes}


def synthesize_entity(types, profiles, quality, target, thresholds) -> dict:
    tc = thresholds["target"]
    plan = {}
    for col, t in types.items():
        prof = profiles.get(col, {})
        q = quality["columns"].get(col, {})
        tgt = (target or {}).get("columns", {}).get(col) if target and target.get("supervised") else None

        rec = _recommend_ops(col, t, prof, q, tgt, thresholds)
        leakage = bool(tgt.get("leakage_suspect")) if tgt else False
        target_signal = ({k: tgt.get(k) for k in
                          ("iv", "iv_band", "auc", "ks", "predictive_rank",
                           "predictive_score") if k in tgt} if tgt else None)

        # priority precedence: drop > unranked > exclude > strong/medium/low
        if q.get("quality_status") == "DROP_SUGGESTED":
            priority = "drop"
        elif not tgt:
            priority = "unranked"
        elif leakage:
            priority = "exclude"
        elif tgt.get("iv_band") == "strong":
            priority = "high"
        elif tgt.get("iv_band") == "medium":
            priority = "medium"
        else:
            priority = "low"

        # Uniform schema: every column emits exactly these keys (null when N/A).
        plan[col] = {
            "column": col,
            "semantic_type": t["semantic_type"],
            "feature_role": rec["feature_role"],
            "recommended_ops": rec["recommended_ops"],
            "requires_human": rec["requires_human"],
            "notes": rec["notes"],
            "quality_status": q.get("quality_status"),
            "target_signal": target_signal,
            "leakage_suspect": leakage,
            "exclude_from_model": leakage,
            "priority": priority,
        }
    return plan


def synthesize_aggregates(relational: dict) -> list:
    """Cross-entity aggregate feature suggestions from the aggregation menus."""
    out = []
    for rel in relational.get("relationships", []):
        parent = rel["to"].split(".")[0]
        child = rel["from"].split(".")[0]
        for cand in rel.get("aggregation_candidates", []):
            for op in cand["ops"]:
                out.append({"parent_entity": parent, "child_entity": child,
                            "join": rel["from"] + " -> " + rel["to"],
                            "feature": f"{child}__{cand['column']}__{op}",
                            "op": op, "source_column": cand["column"]})
    return out



# ============================================================================
# Artifacts (flat table) + markdown report
# ============================================================================

def flatten_columns(entity, types, profiles, quality, target) -> pd.DataFrame:
    """One row per column. `target` is the PRIMARY target result (may be unsup.)."""
    rows = []
    tcols = (target or {}).get("columns", {}) if target and target.get("supervised") else {}
    for col, t in types.items():
        p = profiles.get(col, {})
        q = quality["columns"].get(col, {})
        tg = tcols.get(col, {})
        rows.append({
            "entity": entity, "column": col,
            "semantic_type": t["semantic_type"], "family": t["family"],
            "feature_role": t["feature_role"], "type_source": t["source"],
            "physical_dtype": t["physical_dtype"], "parse_fail_rate": t["parse_fail_rate"],
            "n_present": p.get("n_present"), "pct_missing": p.get("pct_missing"),
            "n_unique": p.get("n_unique"), "cardinality_ratio": p.get("cardinality_ratio"),
            "is_constant": p.get("is_constant"), "dominance": p.get("dominance"),
            "mean": p.get("mean"), "std": p.get("std"), "min": p.get("min"),
            "max": p.get("max"), "skew": p.get("skew"),
            "outlier_frac": (p.get("outliers") or {}).get("fraction"),
            "n_levels": p.get("n_levels"), "true_rate": p.get("true_rate"),
            "validity_rate": p.get("validity_rate"),
            "quality_status": q.get("quality_status"),
            "quality_flags": ";".join(q.get("quality_flags", [])),
            "iv": tg.get("iv"), "iv_band": tg.get("iv_band"), "auc": tg.get("auc"),
            "predictive_rank": tg.get("predictive_rank"),
            "leakage_suspect": tg.get("leakage_suspect"),
        })
    return pd.DataFrame(rows)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_entity_md(entity, catalog, types, profiles, quality, primary, plan,
                     target_results, primary_name=None) -> str:
    sup_targets = [t for t, r in (target_results or {}).items() if r.get("supervised")]
    L = [f"# {entity}", "",
         f"- rows: **{catalog['total_rows']:,}**  ·  columns: **{catalog['n_columns']}**  "
         f"·  shards: **{catalog['n_shards']}**  ·  quality score: **{quality['quality_score']}**",
         f"- mode: **{'supervised' if sup_targets else 'unsupervised'}**"
         + (f"  ·  targets: {', '.join(sup_targets)}" if sup_targets else ""),
         f"- primary target: `{primary_name or ''}`"
         + (f"  (positive_rate={primary.get('positive_rate'):.4f})"
            if primary.get("supervised") else ""),
         ""]
    if catalog.get("schema_conflicts"):
        L.append(f"> ⚠️ schema drift across shards: {len(catalog['schema_conflicts'])} issue(s)")
        L.append("")

    # per-target IV columns
    iv_headers = "".join(f" IV[{t}] |" for t in sup_targets)
    L += ["| column | type | role | missing | unique | quality |" + iv_headers +
          " rank | priority | FE ops |",
          "|---|---|---|---|---|---|" + "|".join(["---"] * len(sup_targets)) +
          ("|" if sup_targets else "") + "---|---|---|"]

    pcols = (primary or {}).get("columns", {})
    cols = sorted(types, key=lambda c: pcols.get(c, {}).get("predictive_rank", 1e9)) \
        if pcols else list(types)
    for col in cols:
        t, p = types[col], profiles.get(col, {})
        q = quality["columns"].get(col, {})
        pl = plan.get(col, {})
        ivs = ""
        for tname in sup_targets:
            r = target_results[tname].get("columns", {}).get(col, {})
            band = r.get("iv_band")
            leak = " ⚠" if r.get("leakage_suspect") else ""
            ivs += f" {_fmt(r.get('iv'))}{f' ({band})' if band else ''}{leak} |"
        ops = ", ".join(o.get("op", "?") for o in pl.get("recommended_ops", []))
        if pl.get("requires_human"):
            ops += " **(human)**"
        prank = pcols.get(col, {}).get("predictive_rank", "")
        L.append(f"| {col} | {t['semantic_type']} | {t['feature_role']} | "
                 f"{(p.get('pct_missing') or 0):.1%} | {p.get('n_unique')} | "
                 f"{q.get('quality_status')} |{ivs} {prank} | "
                 f"{pl.get('priority','')} | {ops} |")

    # quality call-outs
    drops = [c for c in types if quality["columns"].get(c, {}).get("quality_status") == "DROP_SUGGESTED"]
    if drops:
        L += ["", f"**Drop suggested ({len(drops)}):** " + ", ".join(f"`{c}`" for c in drops)]
    return "\n".join(L) + "\n"


def write_report(report_dir, name, md):
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{name}.md"), "w") as f:
        f.write(md)
