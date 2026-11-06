# EDA Pipeline — Current State, Data Flow, and Roadmap

> Companion to `EDA_PROCEDURE.md` (the spec) and `README.md` (the quickstart).
> This document explains **how the engine works end-to-end today**, **what it
> already computes**, **what is still missing**, and a **phased plan** to make EDA
> comprehensive — with **feature generation deliberately pushed to the very end**.

---

## 0. The big picture in one diagram

```
                         ┌────────────────────── USER INPUTS (config/) ──────────────────────┐
                         │ datasource.yaml   target.yaml      thresholds.yaml                  │
                         │ (what & where)    (labels)         (decision cutoffs)               │
                         └─────────────────────────────────┬─────────────────────────────────┘
                                                           │
   RAW DATA (parquet shards)                               ▼
   pnb_mapped/                          ┌─────────────────────────────────────┐
     accounts/*.parquet      ─────────► │              EDA ENGINE             │
     customers/*.parquet               │   Setup → Analyze → Score → Output   │
     transactions/*.parquet            └───────────────────┬─────────────────┘
     mha_l0/*.parquet  (victim label)                      │
     mha_l1/*.parquet  (fraudster label)                   │
                                                            ▼
                         ┌──────────────── OUTPUTS ────────────────┐
                         │ artifacts/ (JSON + parquet, machine)     │  ─► Feature-Engineering step (LATER)
                         │ report/    (markdown, human)             │  ─► Humans review & sign off
                         └──────────────────────────────────────────┘
```

**Two rules that make it data-agnostic** (never break these):
1. Code branches only on `semantic_type`, **never on column names**. All naming lives in `config/`.
2. Every metric is optional & self-describing — if it can't be computed it's omitted with a `reason`, never an exception.

---

## 1. From raw data to user inputs — what the user actually controls

The user touches **only three config files**. Nothing about the data is in code.

| File | The user declares… | Example |
|---|---|---|
| `datasource.yaml` | where data lives (`root`, globs), entity grain, primary/foreign keys, time column, per-column `semantic_type` overrides, sampling caps | `TRANSACTION_AMOUNT: {semantic_type: amount}` |
| `target.yaml` | the label(s): which entity membership = positive, how to join, which feature entities the label applies to (with optional bridge `via:`) | `is_fraudster_account` ← present in `mha_l1` |
| `thresholds.yaml` | every decision cutoff: missing %, IV bands, outlier method, cardinality limits, sampling sizes, report caps | `iv_strong: 0.3` |

**Current dataset (PNB fraud):** 5 entities, 2 labels.

| Entity | Grain | Role | Rows (this run) | Cols |
|---|---|---|---|---|
| `accounts` | one bank account | feature source | 668,884 | 48 |
| `customers` | one customer | feature source | 673,969 | 32 |
| `transactions` | one transaction | feature source | 128,048 | 26 |
| `mha_l0` | one complaint | **victim label** | 119,843 | 4 |
| `mha_l1` | one fraud-txn mapping | **fraudster label** | 208,572 | 7 |

Two targets, kept **separate** on purpose: `is_fraudster_account` (primary, from `mha_l1`) and `is_victim_account` (from `mha_l0`). Victim ≠ fraudster; never merged.

---

## 2. The data-type system (the brain of agnosticism)

Physical dtype (`int64`, `string`, `timestamp`) is **not enough** — `PINCODE` is an int but geographic-categorical; `ANNUAL_INCOME` is a number stored as a string; `*_FLAG` strings are booleans. So every column is assigned a **`semantic_type`**, and everything downstream branches on its **family**.

| Family | semantic_types | Analyses applied |
|---|---|---|
| `numeric` | continuous, amount, count, amount_string | distribution stats, outliers, skew, IV via quantile bins, AUC/KS/point-biserial |
| `categorical` | categorical, ordinal, boolean_flag, binary, categorical_id | frequency table, imbalance, IV/WoE, Cramér's V |
| `datetime` | datetime | span, granularity, future/sentinel dates, recency stubs |
| `pattern` | ip_address, email, vpa, bank_code, phone, pincode | regex validity rate, decomposition stubs, IV (capped) |
| `keylike` | identifier, hashed_pii | uniqueness/dup checks; **not a model feature** (join only) |
| `text` | free_text | length/token stats; vectorization **deferred to human/LLM** |
| `other` | constant, unknown | universal profile only; `unknown` flagged for a human |

Resolution order: **config override wins**, else inferred from a sample via the ladder in `typing_registry.infer_semantic_type`. If a coercion (cast) fails for >50% of values, the type is **downgraded** (e.g. bad numeric → categorical/text) and flagged — never an error.

---

## 3. How the engine runs today — the 4 stages in detail

Each stage **appends** to a growing per-column record; the collection of records **is** the contract.

### Stage 1 — SETUP (`setup.py`)  ·  raw files → catalog
1. **Resolve shards** from each glob, natural-sorted (`2_RBIH` before `10_RBIH`).
2. **Read parquet schema only** (cheap metadata): per-shard row count, bytes, columns, arrow types.
3. **Schema reconciliation across shards** → union of columns; flag columns missing in some shards and dtype mismatches (`schema_conflicts`).
4. **Load plan**: `full` if total rows ≤ `profile_row_cap` (2M), else stratified `sample`.
5. Emit `catalog.json`.

### Stage 2 — ANALYZE (`analyze.py`)  ·  the per-column work
- **Semantic typing** (`type_entity`): override→infer→coerce→fail-soft downgrade; records `parse_fail_rate`, `feature_role` (key vs feature), PK/FK flags.
- **Univariate profiling** (`profile_entity`), conditioned on family:
  - *Universal (every column):* count, missing %, n_unique, cardinality ratio, blanks, top-k values, dominance, constant/near-constant flags, Shannon + normalized entropy.
  - *Numeric:* min/max/range, mean/trimmed-mean/median, std/var/MAD/IQR/CV, skew/kurtosis + shape tag, percentiles (1–99), zero-inflation, sign profile, integer check, outlier fraction (IQR/z/MAD), 30-bin histogram, log-transform candidacy.
  - *Categorical:* level count, max-class share, Gini, rare-level count & mass, case/whitespace variant collapse, imbalance.
  - *Boolean:* true-rate, n_true/n_false. *Ordinal:* + order coverage (missing/unexpected levels).
  - *Datetime:* parse rate, min/max/span, unique days, time-component detection, future & sentinel dates, by-year/dow/hour, derived-feature stubs (recency/age/tenure).
  - *Pattern:* regex validity rate, invalid examples, decomposition stubs (subnet, PSP handle, bank code…).
  - *Text:* char/token length, whitespace ratio, embedded-digit fraction (vectorization deferred).
  - *Keylike:* duplicate count, uniqueness ratio, length consistency.
- **Quality scorecard** (`quality_entity`): per-column status `OK|WARN|DROP_SUGGESTED` + machine-readable **remediation actions** (drop/impute/cast/outlier_cap/standardize…); per-entity PK uniqueness, full-row dup rate, schema conflicts, and a single **quality score** = `100·(1 − (drops + 0.4·warns)/n_cols)`.
- **Within-entity relationships** (`multivariate_entity`): Pearson matrix → top pairs + redundancy clusters (|r|≥0.95); Cramér's V categorical associations; functional dependencies (A→B).
- **Temporal split suggestion** (`drift_entity`): the entity's time column's **80th-percentile timestamp** as a leakage-safe train/valid cut. (Cross-shard drift is intentionally *not* computed — shards are arbitrary size-splits of one population, not time batches.)

### Stage 3 — SCORE (`score.py`)  ·  target-aware + cross-entity
- **Attach label** (`label_series`): map each feature row to 0/1 by membership in the label entity; bridge through a parent when `via:` is set (e.g. customers→accounts→mha_l1). Records `positive_rate`, `n_pos`, `n_neg`.
- **Per-column predictive power**, computed **separately per target**:
  - *Numeric:* IV over **10 quantile bins**, single-feature **AUC**, **KS**, **point-biserial**.
  - *Categorical/pattern:* IV/WoE (one bin per value; high-card columns capped to **top-10 + `__other__`**), **Cramér's V** vs target, target-rate-by-category.
- **Rank** all feature columns by a combined score (max of IV, rescaled AUC, |point-biserial|) → `predictive_rank` (vs the **primary** target).
- **Leakage guard**: flag columns with AUC ≥ 0.995 or IV > 5 as `leakage_suspect` → excluded from auto-FE.
- **Cross-entity relational** (`relational_eda`): per FK, classify cardinality (N:1 / N:M), **orphan rate**, **fan-out stats** (children per parent), childless-parent rate, and an **aggregation menu** (which child columns roll up to the parent grain, and how).

### Stage 4 — OUTPUT (`output.py`)  ·  synthesis & rendering
- **Feature-plan synthesis** (`synthesize_entity`): deterministically map each column's profile → recommended FE ops; set `priority` (drop > unranked > exclude > high/medium/low), `requires_human`, `leakage_suspect`, `target_signal`.
- **Cross-entity aggregate suggestions** (`synthesize_aggregates`): expand the aggregation menu into concrete feature names (e.g. `transactions__TRANSACTION_AMOUNT__sum` per account) — **234 suggestions** in this run.
- **Flat table** (`flatten_columns` → `columns.parquet`): one row per column, queryable in SQL/pandas.
- **Markdown report** (`render_entity_md`): the per-entity table you've been reading.

---

## 4. How data is divided (sampling, joins, splits)

This is the part that's easy to miss, so it's spelled out:

| Division | Where | What happens | Knob |
|---|---|---|---|
| **Sharding** | Setup | Each entity = many parquet files, unioned by column | `glob` |
| **Row-cap sampling** | `io.load_entity` | If total rows > `profile_row_cap` (2M), load a **per-shard stratified sample** so every number is interpretable; the run records `sampled: true` | `sampling.profile_row_cap`, `per_shard_stratify` |
| **Heavy-op sampling** | profiling/multivariate/score | O(n²)/pattern/corr/IV ops run on a `heavy_op_sample` (200K) subset, seeded | `sampling.heavy_op_sample` |
| **Label join** | Score | Positives = membership in label entity; bridge `via:` a parent if keys don't align directly | `target.yaml applies_to` |
| **Class split** | Score | Rows split into positive (fraud/victim) vs negative for IV/WoE/AUC | `default_negative: true` |
| **Time split (suggested only)** | Analyze | 80th-percentile timestamp recorded as a train/valid cut — **not enforced** | `time_column` |

> ⚠️ **Note on `transactions` in this run:** only 128K rows and `ACCOUNT_NUMBER` has just **200 distinct values** with ~640 txns/account. This entity looks like a *narrow subset* (a few accounts), so its IVs rest on a small account base — interpret transaction-level signal with care until the full table is profiled.

---

## 5. What results are calculated (the full inventory)

**Per column** (lands in `profile.json`, `feature_plan.json`, `columns.parquet`):
type · family · feature_role · parse_fail_rate · completeness · cardinality · entropy · (numeric: full distribution + outliers + histogram) · (categorical: frequencies + imbalance + rare levels) · (datetime: span + anomalies) · (pattern: validity) · quality status + flags + actions · **per-target IV / IV-band / AUC / KS / point-biserial / Cramér's V / target-rate-by-cat** · predictive rank · leakage flag · priority · recommended FE ops.

**Per entity** (`profile.json`, `drift.json`, `target.json`):
row/column counts · schema conflicts · PK uniqueness · full-row dup rate · **quality score** · within-entity correlations / associations / functional dependencies · positive rate per target · suggested time split.

**Cross entity** (`relational.json`, global `feature_plan.json`):
join cardinality · orphan rate · fan-out distribution · aggregation menu · 234 aggregate-feature suggestions.

---

## 6. Gap analysis — what's done vs what's missing

### ✅ Strong / done
- Robust semantic typing with override + inference + fail-soft downgrade.
- Comprehensive univariate profiling per family.
- Quality scorecard with machine-readable remediation actions + entity score.
- Within-entity correlation / Cramér's V / functional dependencies.
- Dual-target IV/WoE/AUC/KS/point-biserial, kept separate, high-card capped, leakage-guarded.
- Cross-entity join characterization + aggregation menu.
- Graceful degradation everywhere (never crashes the orchestrator).

### ⚠️ Documented-but-not-implemented (spec/README claims it; code doesn't do it)
- **Mutual Information (MI)** — listed in README/spec ("IV/WoE/AUC/KS/MI") but **no MI function exists** and Score never computes it.
- **`woe_min_bin_frac` (≥5% per bin) + monotonic binning** — config knob exists but numeric IV uses plain `qcut(10, duplicates='drop')`; the min-bin-fraction is **never enforced** and binning is **not monotonic**.
- **PSI / distribution drift** — `metrics.psi` is implemented but **unused**; only a time-split *timestamp* is produced, no actual drift measured.
- **`correlation_ratio` (eta, numeric↔categorical)** — implemented but **never called**; numeric-vs-categorical relationships aren't analyzed.
- **Interaction candidates** — spec mentions flagging pairs for FE crosses; not implemented beyond raw associations.

### ❌ Genuinely missing (worth adding for "comprehensive")
- **No enforced train/valid/test partition** — only a suggested cut; no leakage-safe split is persisted or used.
- **No missingness analysis** — no missingness correlation / patterns / MAR-vs-MCAR signal.
- **No datetime target signal** — datetime columns are never scored against the label (no recency-vs-fraud, seasonality-vs-fraud).
- **No target analysis for derived/temporal features** — only raw columns scored.
- **Shallow leakage detection** — only an AUC/IV threshold; no systematic "is this column downstream of the label / sourced from `mha_*`" tracing.
- **No class-imbalance treatment guidance** beyond reporting `positive_rate`.
- **No bivariate target visuals / calibration / lift tables** in the report.
- **Aggregate features are only *suggested*, never *computed*** — by design today (and we want to keep FE late), but there's no validation that the suggestions are sound.
- **No deduplication / entity-resolution** across accounts↔customers↔transactions.
- **No data dictionary / lineage** output tying every emitted artifact field back to its source.

---

## 7. The vision: comprehensive EDA, feature generation **last**

You want EDA to be deep and *understanding-first*, with feature creation pushed to the very end. The current engine already separates **"emit a plan"** (Output stage) from **"execute the plan"** (a later FE step) — but it mixes *feature recommendation* into the EDA run. The proposed restructure makes the boundary explicit and inserts the missing analyses **before** any feature thinking.

### Proposed phase model (EDA ends at Phase 6; FE is Phases 7–9)

```
─────────────────────────  EDA (understanding only)  ─────────────────────────
 Phase 0  INGEST & CATALOG      shards, schema, drift, load plan        [DONE]
 Phase 1  TYPE & COERCE         semantic types, coercion, fail-soft     [DONE]
 Phase 2  PROFILE (univariate)  full per-family stats                   [DONE]
 Phase 3  QUALITY & SANITY      scorecard, actions, entity score        [DONE]
 Phase 4  RELATIONSHIPS         within + CROSS entity, +eta, +MI,       [PARTIAL]
                                +interactions, +missingness patterns
 Phase 5  TARGET & DRIFT        IV/WoE/AUC/KS + MI, monotonic bins,      [PARTIAL]
                                datetime signal, PSI drift, deeper
                                leakage tracing
 Phase 6  PARTITION & SIGN-OFF  enforce leakage-safe train/valid/test,  [MISSING]
                                imbalance report, human review GATE
────────────────────────────  ← EDA STOPS HERE →  ────────────────────────────
 Phase 7  FEATURE SPEC          synthesize the plan (today's Output)    [DONE, move later]
 Phase 8  FEATURE GENERATION    actually materialize columns +          [MISSING — the "very later" step]
                                cross-entity aggregates
 Phase 9  FEATURE VALIDATION    re-score generated features, drift,     [MISSING]
                                leakage re-check, dedup vs raw
```

Key principle: **Phases 0–6 never invent a feature.** They only *describe and partition* the data and produce a human sign-off gate. The feature plan (Phase 7) and especially generation (Phase 8) consume the EDA contract but run as a distinct, later stage — exactly the "feature generation comes very later" design you want.

---

## 8. Concrete next steps (prioritized)

**Tier 1 — close the "documented but missing" gaps (fast, high trust payoff)**
1. Implement **MI** (`mutual_info_classif`-style) in `metrics/` and wire into Score, or remove the claim from README/spec.
2. Enforce **`woe_min_bin_frac`** and add **monotonic binning** for numeric IV.
3. Either **use `psi`** for a real drift signal (e.g. train-vs-valid post-split) or document that drift = time-split only.
4. **Call `correlation_ratio`** to add numeric↔categorical relationships.

**Tier 2 — make EDA comprehensive (the understanding layer)**
5. **Missingness analysis**: missingness correlation matrix, co-missing clusters, missingness-vs-target signal.
6. **Datetime target signal**: recency / tenure / hour-of-day scored against each label.
7. **Deeper leakage tracing**: tag any column lineage-connected to `mha_*` as leakage by construction, not just by AUC.
8. **Interaction candidates**: surface high-association pairs as FE-cross suggestions (suggest only).

**Tier 3 — partition & gate (Phase 6)**
9. **Enforce a leakage-safe split** (time-based where a time column exists, stratified otherwise) and persist `splits.json`.
10. **Imbalance report** + recommended handling (class weights / sampling) — recorded, not applied.
11. **Human sign-off gate**: a single `eda_signoff.md` summarizing entities, quality scores, top signals, leakage suspects, and open `requires_human` items.

**Tier 4 — feature generation, last (Phases 7–9)**
12. Keep **feature *spec*** (today's Output) but relabel it as a post-EDA stage.
13. Add a **feature *generation*** module that materializes per-column ops + the 234 cross-entity aggregates into an actual feature table.
14. **Re-run EDA scoring on generated features** (leakage re-check, drift, dedup vs raw) before modeling.

---

## 9. TL;DR

- **Today:** a solid, data-agnostic, no-LLM EDA engine that types, profiles, quality-checks, relates, target-scores (dual labels), and characterizes cross-entity joins — then emits a machine-readable feature plan + markdown report. Stages 1–3 of "understanding" are largely done.
- **Missing for "comprehensive":** MI, monotonic/min-fraction binning, real drift (PSI), numeric↔categorical (eta), missingness analysis, datetime target signal, deeper leakage tracing, and an **enforced leakage-safe partition + human sign-off gate**.
- **Your design goal (feature generation last) fits cleanly:** make Phases 0–6 pure understanding + partitioning, end EDA at a human sign-off gate, and run feature **spec → generation → validation** as distinct later phases that only consume the EDA contract.
</content>
</invoke>
