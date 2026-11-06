# Data-Agnostic EDA Engine

Automated, config-driven, **no-LLM** EDA stage for the model-training orchestrator.
Profiles every column of every entity, runs sanity checks, measures
structure/drift, ranks predictive power against configured labels, and emits a
**machine-readable feature plan** the downstream feature-engineering step executes.

Full methodology: [`EDA_PROCEDURE.md`](./EDA_PROCEDURE.md).

## Run

```bash
python -m eda_engine.orchestrator --config ./config                 # all entities
python -m eda_engine.orchestrator --config ./config --entities accounts,transactions
python -m eda_engine.orchestrator --config ./config --no-report     # artifacts only
```

## Where data knowledge lives (and ONLY here)

| File | Holds |
|---|---|
| `config/datasource.yaml` | entities, globs, keys, foreign keys, column type overrides |
| `config/target.yaml`     | label definitions (here: `is_fraudster_account` from `mha_l1`, `is_victim_account` from `mha_l0`) |
| `config/thresholds.yaml` | all decision cutoffs (missing %, IV bands, PSI, outliers, sampling) |

Code under `eda/` branches only on `semantic_type` — never on column names. Point
the config at a different dataset and the same engine runs.

## Outputs

```
artifacts/
  catalog.json                 # schemas, row counts, schema drift, load plan
  relational.json              # cross-entity joins + aggregate feature suggestions
  feature_plan.json            # GLOBAL contract (per-entity plan files + aggregates)
  <entity>/
    profile.json               # types + univariate profile + quality rollup + within-entity relationships
    drift.json                 # suggested train/valid time split
    target.json                # per-target IV/AUC/KS + leakage flags (high-card capped)
    feature_plan.json          # per-column recommended FE ops (uniform schema)
    columns.parquet            # FLAT one-row-per-column table (query in SQL/pandas)
report/
  <entity>.md                  # human-viewable markdown, one per entity
```

The feature-engineering step loads `feature_plan.json` (+ per-entity plans),
executes every op where `requires_human == false`, and surfaces the rest to a
person. EDA itself stays 100% deterministic and LLM-free.

## Stages
The engine runs in 4 stages, one module each:

| Module | Does |
|---|---|
| `setup.py`   | config load/validate + run manifest; catalog (schemas, row counts, schema drift, load plan) |
| `analyze.py` | semantic typing · univariate profiling · quality scorecard+actions · within-entity relationships · drift |
| `score.py`   | target-aware IV/WoE/AUC/KS/MI + leakage guard; cross-entity joins + aggregation menu |
| `output.py`  | feature-plan synthesis; flat columns table + markdown report |

`orchestrator.py` runs the 4 stages in order. Shared, data-agnostic helpers live
in `utils/` (`io.py`, `typing_registry.py`, `metrics/`).
