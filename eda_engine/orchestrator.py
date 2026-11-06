"""Orchestrator -- runs the 4 stages (Setup, Analyze, Score, Output) per entity.
CLI entrypoint.

Usage:
    python -m eda_engine.orchestrator --config ./config [--entities accounts,customers] [--no-report]
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

from . import setup, analyze, score, output
from .utils import io


def run(config_dir: str, only_entities=None, make_report=True) -> dict:
    t0 = time.time()
    cfg = setup.load_config(config_dir)
    ds = cfg["datasource"]
    th = cfg["thresholds"]
    th["_heavy_sample"] = ds["sampling"]["heavy_op_sample"]
    sampling = ds["sampling"]
    out_dir = ds["output_dir"]
    report_dir = ds["report_dir"]

    # resolve ALL entities (label/bridge resolution must see every entity,
    # even when --entities profiles only a subset).
    full_resolved = setup.resolve_entities(cfg)
    resolved = {"entities": dict(full_resolved["entities"]),
                "skipped": full_resolved["skipped"]}
    if only_entities:
        keep = set(only_entities)
        resolved["entities"] = {k: v for k, v in resolved["entities"].items() if k in keep}

    # catalog (only profiled subset)
    catalog = setup.build_catalog(resolved, sampling)
    io.write_json(os.path.join(out_dir, "catalog.json"), catalog)

    # resolve ALL active targets against the FULL entity set
    tctx_map = score.resolve_targets(cfg, full_resolved)
    primary_name = score.primary_target_name(cfg)
    available_targets = {n: c for n, c in tctx_map.items() if c.get("available")}
    supervised_available = bool(available_targets)

    types_by_entity, scores = {}, {}
    for name, ecfg in resolved["entities"].items():
        print(f"[{name}] loading...", flush=True)
        try:
            df, load_info = io.load_entity(
                ecfg["shards"], row_cap=sampling["profile_row_cap"],
                per_shard_stratify=sampling["per_shard_stratify"],
                seed=sampling["random_seed"])
        except Exception as e:
            io.write_json(os.path.join(out_dir, name, "ERROR.json"),
                          {"stage": "load", "error": str(e), "trace": traceback.format_exc()})
            continue

        # semantic typing
        types = analyze.type_entity(df, ecfg, th)
        types_by_entity[name] = types

        # univariate profiling
        profiles = analyze.profile_entity(df, types, ecfg, th)

        # quality
        quality = analyze.quality_entity(df, types, profiles, catalog[name], th)

        # multivariate
        try:
            multivar = analyze.multivariate_entity(df, types, th)
        except Exception as e:
            multivar = {"_error": str(e)}

        # drift
        try:
            drift = analyze.drift_entity(ecfg["shards"], types, catalog[name], th)
        except Exception as e:
            drift = {"_error": str(e)}

        # target-aware -- analyze EVERY available target (skip-safe per target)
        target_results = {}
        for tname, tctx in available_targets.items():
            y = score.label_series(name, df, tctx, full_resolved)
            if y is not None:
                target_results[tname] = score.target_entity(df, types, y, th)
            else:
                target_results[tname] = {"supervised": False,
                                         "reason": "target not applicable to entity"}
        # primary target drives feature-plan priority / report ordering
        primary = target_results.get(primary_name) or \
            next((r for r in target_results.values() if r.get("supervised")), None) or \
            {"supervised": False, "reason": "no target applicable"}
        target = {"primary_target": primary_name, "results": target_results,
                  **{k: primary[k] for k in ("supervised", "reason") if k in primary}}

        # feature-plan synthesis (uses primary target signal)
        plan = output.synthesize_entity(types, profiles, quality, primary, th)

        # persist machine-readable artifacts
        ent_dir = os.path.join(out_dir, name)
        # profile.json is the per-entity home: typing + univariate profiles, the
        # entity-level quality rollup, and within-entity relationships. Per-column
        # quality verdicts already flow into feature_plan.json + columns.parquet.
        io.write_json(os.path.join(ent_dir, "profile.json"),
                      {"types": types, "profiles": profiles, "load_info": load_info,
                       "quality_score": quality["quality_score"],
                       "entity_checks": quality["entity_checks"],
                       "relationships": multivar})
        io.write_json(os.path.join(ent_dir, "drift.json"), drift)
        io.write_json(os.path.join(ent_dir, "target.json"), target)
        io.write_json(os.path.join(ent_dir, "feature_plan.json"), plan)
        flat = output.flatten_columns(name, types, profiles, quality, primary)
        io.write_parquet(os.path.join(ent_dir, "columns.parquet"), flat)

        scores[name] = quality["quality_score"]

        if make_report:
            md = output.render_entity_md(name, catalog[name], types, profiles,
                                             quality, primary, plan, target_results,
                                             primary_name)
            output.write_report(report_dir, name, md)
        print(f"[{name}] done. quality={quality['quality_score']} "
              f"supervised={primary.get('supervised')} "
              f"targets={[t for t,r in target_results.items() if r.get('supervised')]}",
              flush=True)

    # cross-entity relational (use full entity set for join analysis)
    relational = score.relational_eda(full_resolved, types_by_entity, th)
    agg_feats = output.synthesize_aggregates(relational)
    relational["aggregate_feature_suggestions"] = agg_feats
    io.write_json(os.path.join(out_dir, "relational.json"), relational)

    # global feature_plan = per-entity plans + cross-entity aggregates
    global_plan = {"entities": list(types_by_entity), "supervised": supervised_available,
                   "primary_target": primary_name,
                   "active_targets": list(available_targets),
                   "aggregate_feature_suggestions": agg_feats,
                   "per_entity_plan_files": {e: f"artifacts/{e}/feature_plan.json"
                                             for e in types_by_entity}}
    io.write_json(os.path.join(out_dir, "feature_plan.json"), global_plan)

    elapsed = round(time.time() - t0, 1)
    print(f"\nEDA complete in {elapsed}s. artifacts -> {out_dir}  report -> {report_dir}")
    return {"elapsed_s": elapsed, "scores": scores,
            "supervised": supervised_available, "n_entities": len(types_by_entity)}


def main():
    ap = argparse.ArgumentParser(description="Data-agnostic EDA engine")
    ap.add_argument("--config", default="./config")
    ap.add_argument("--entities", default=None, help="comma-separated subset")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    only = args.entities.split(",") if args.entities else None
    run(args.config, only_entities=only, make_report=not args.no_report)


if __name__ == "__main__":
    main()
