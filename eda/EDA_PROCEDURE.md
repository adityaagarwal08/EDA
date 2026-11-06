# Data-Agnostic EDA Procedure (Spec)

> Purpose: a fully automated, **no-LLM**, **config-driven** EDA stage inside a job
> orchestrator. It profiles **every column of every entity**, runs sanity checks,
> measures structure/relationships/drift, and — when a label is available —
> ranks predictive power. Its **output is a machine-readable contract** that the
> downstream Feature-Engineering (FE) step consumes to **auto-generate features**,
> plus a **human-viewable report**.
>
> Hard constraints (from requirements):
> - **Pandas** compute engine.
> - **Nothing data-specific in code** — all data knowledge lives in `config/`.
> - **Both** unsupervised + target-aware; if no label is configured/available for
>   an entity (e.g. `mha_l0`/`mha_l1` absent), **silently fall back to unsupervised,
>   never error**.
> - **Comprehensive** per-column depth.
> - Results are **persisted** (artifacts) **and shown** (report).
> - No human in the loop for EDA; FE features that need domain knowledge are
>   **flagged** for a human, not invented.

---

## 0. Mental model

```
                 ┌──────────────┐
   config/  ───► │   EDA ENGINE │ ───►  artifacts/  (JSON + parquet)  ──► FE step
   (all data     │   4 stages   │ ───►  report/     (markdown)         ──► humans
    knowledge)   └──────────────┘
```

The engine runs in **four stages** — **Setup → Analyze → Score → Output**. Each
stage reads the previous stages' outputs and **appends** to a single growing
per-column record. The final record for one column is the EDA "passport" of that
column. The collection of passports **is** the contract.

Two unbreakable rules make it data-agnostic:

1. **Code branches on `semantic_type`, never on column names.** Column names,
   domains, valid value sets, label definitions, join keys → all in config.
2. **Every metric is optional and self-describing.** If a metric can't be computed
   (wrong type, no label, too few rows), it is omitted with a recorded `reason`,
   never an exception.

---

## 1. Entities, the running example

The pipeline treats data as a set of **entities**, each possibly **sharded** into
multiple files. Using `pnb_mapped/` as the worked example:

| Entity         | Files (shards)              | Grain (1 row =)        | Key(s)                          | Role           |
|----------------|-----------------------------|------------------------|---------------------------------|----------------|
| `accounts`     | `ACCOUNT_DATA_part_{1..3}`  | one bank account       | `ACCOUNT_NUMBER` → `CUSTOMER_ID`| feature source |
| `customers`    | `CUST_DATA_part_{1..3}`     | one customer           | `CUSTOMER_ID`                   | feature source |
| `transactions` | `{1..10}_RBIH_DATA`         | one transaction        | `TRANSACTION_ID`, FK `ACCOUNT_NUMBER`,`CUSTOMER_ID` | feature source |
| `mha_l0`       | `L0.parquet`                | one complaint          | `ACKNOWLEDGEMENT_NUMBER`, FK `ACCOUNT_NUMBER` | **label source** |
| `mha_l1`       | `L1.parquet`                | one fraud txn mapping  | `TXN_ID`, `FRAUD_LAYER`, FK `ACCOUNT_NUMBER` | **label source** |

Nothing above is hard-coded; it is exactly the shape of `config/datasource.yaml`
(Section 3). Point the config at a different folder and the same engine runs.

---

## 2. The stages (overview)

| Stage | Module | Reads | Produces |
|-------|--------|-------|----------|
| **Setup**   | `setup.py`   | `config/` + files | validated run-config & manifest; data catalog (shards, schemas, row counts, schema drift, load plan) |
| **Analyze** | `analyze.py` | catalog + loaded columns | per-column `semantic_type` + coercion plan; univariate statistics; quality scorecard + actions; within-entity relationships; drift/PSI |
| **Score**   | `score.py`   | label joins + keys across entities | IV/WoE/AUC/KS/MI ranked predictive power + leakage flags; join cardinality, orphan rates, aggregation map |
| **Output**  | `output.py`  | all of the above | **feature plan** (the FE contract); JSON/parquet artifacts + markdown report |

Setup and Output bookend the run; Analyze and Score do the per-entity work
(Score's relational pass spans entities). Each stage is independently
re-runnable and its results cache to `artifacts/<entity>/*.json`.

---

# Stage 1 — Setup

## 3. Config & contracts (where ALL data knowledge lives)

Three config files. Anything that "knows about the data" goes here.

### 3.1 `config/datasource.yaml` — what & where

```yaml
root: /home/ai-centre-of-excellence/aditya/pnb_mapped
engine: pandas
output_dir: ./artifacts
report_dir: ./report

# Global sampling/perf knobs (Pandas → cost matters)
sampling:
  profile_row_cap: 2_000_000      # cap rows loaded per entity for heavy passes
  heavy_op_sample: 200_000        # rows for O(n^2)/pattern/corr ops
  random_seed: 42
  per_shard_stratify: true        # sample evenly across shards for drift fairness

entities:
  accounts:
    glob: "accounts/*.parquet"
    grain: "one account"
    primary_key: [ACCOUNT_NUMBER]
    foreign_keys:
      CUSTOMER_ID: { references: customers.CUSTOMER_ID }
    # OPTIONAL per-column hints (override inference). Omit to auto-infer.
    column_overrides:
      PINCODE:        { semantic_type: pincode }
      BRANCH_CODE:    { semantic_type: categorical_id }
      "*_FLAG":       { semantic_type: boolean_flag, true_values: [Y, "1", TRUE] }
      LIEN_AMOUNT:    { semantic_type: amount, currency: INR }
      "*_DATE":       { semantic_type: datetime }
      KYC_RATING:     { semantic_type: ordinal, order: [LOW, MEDIUM, HIGH] }

  customers:
    glob: "customers/*.parquet"
    primary_key: [CUSTOMER_ID]
    column_overrides:
      "HASH_OF_*":    { semantic_type: hashed_pii }     # high-card, do NOT one-hot
      DATE_OF_BIRTH:  { semantic_type: datetime, derive_age: true }
      ANNUAL_INCOME:  { semantic_type: amount_string }  # numeric stored as string
      ADDRESS:        { semantic_type: free_text }
      "*_BALANCE":    { semantic_type: amount, currency: INR }
      "NUM_*":        { semantic_type: count }

  transactions:
    glob: "transactions/*.parquet"
    primary_key: [TRANSACTION_ID]
    foreign_keys:
      ACCOUNT_NUMBER: { references: accounts.ACCOUNT_NUMBER }
      CUSTOMER_ID:    { references: customers.CUSTOMER_ID }
    column_overrides:
      TRANSACTION_DATE: { semantic_type: datetime, format: infer }
      TRANSACTION_TIME: { semantic_type: datetime, format: infer }
      TRANSACTION_AMOUNT: { semantic_type: amount, currency: INR }
      INITIATOR_IP_ADDRESS: { semantic_type: ip_address }
      MOBILE_NUMBER:    { semantic_type: phone }
      "BENEFICIARY_*":  {}      # let inference decide
      "REMITTER_*":     {}
      "*_IFSC":         { semantic_type: bank_code }
      "*_VIRTUAL_ADDRESS": { semantic_type: vpa }

  mha_l0:
    glob: "mha_l0/*.parquet"
    primary_key: [ACKNOWLEDGEMENT_NUMBER]
    optional: true              # ← if folder missing, skip without error
    column_overrides:
      COMPLAINT_DATE: { semantic_type: datetime }
      DISPUTED_AMOUNT:{ semantic_type: amount, currency: INR }

  mha_l1:
    glob: "mha_l1/*.parquet"
    primary_key: [TXN_ID]
    optional: true
    column_overrides:
      FRAUD_LAYER:  { semantic_type: ordinal, order: [0,1,2,3] }
      TXN_AMOUNT:   { semantic_type: amount_string }
      DISPUTED_AMOUNT: { semantic_type: amount_string }
```

### 3.2 `config/target.yaml` — the label(s) (optional; drives the Score stage)

This is the **only** place labels are defined. If this file is absent, or a
label's source entities are missing, that target is skipped and (if no target
resolves) the run is **unsupervised** — never an error.

> **Domain fact (from data owner):** `mha_l0` = **victim** accounts,
> `mha_l1` = **fraudster** accounts. These are *different* labels and must **not**
> be merged. So we define **two** targets and the Score stage ranks every column
> against **each** one separately.

```yaml
targets:
  is_fraudster_account:                 # primary detection goal
    type: binary
    positive_if_present_in:
      - entity: mha_l1                  # fraudster / mule accounts
        join: { left: ACCOUNT_NUMBER, right: ACCOUNT_NUMBER }
        # optional: filter: "FRAUD_LAYER >= 1"  to narrow the definition
    applies_to:
      accounts:     { key: ACCOUNT_NUMBER }
      transactions: { key: ACCOUNT_NUMBER }
      customers:    { key: CUSTOMER_ID, via: accounts }   # bridge through accounts
    default_negative: true             # rows not matched → 0 (not null)

  is_victim_account:                    # secondary label
    type: binary
    positive_if_present_in:
      - entity: mha_l0                  # victim accounts
        join: { left: ACCOUNT_NUMBER, right: ACCOUNT_NUMBER }
    applies_to:
      accounts:     { key: ACCOUNT_NUMBER }
      transactions: { key: ACCOUNT_NUMBER }
      customers:    { key: CUSTOMER_ID, via: accounts }
    default_negative: true

active_targets: [is_fraudster_account, is_victim_account]  # both analyzed
primary_target: is_fraudster_account    # drives feature-plan priority & report order
```

### 3.3 `config/thresholds.yaml` — knobs, not data

All decision cutoffs live here so behavior is tunable without code edits:

```yaml
quality:
  missing_warn: 0.20            # flag column if >20% missing
  missing_drop_suggest: 0.60   # suggest drop if >60% missing
  near_constant_dominance: 0.99 # top value covers ≥99%
  high_cardinality_ratio: 0.90  # unique/non-null ≥0.9 → likely ID/free text
  id_uniqueness_min: 0.98       # candidate primary key
  outlier_method: iqr           # iqr | zscore | mad
  outlier_iqr_k: 1.5
type_inference:
  categorical_max_unique: 50    # ≤50 distinct → categorical (else high-card)
  categorical_max_ratio: 0.05
  datetime_parse_success_min: 0.90
target:
  iv_weak: 0.02                 # IV bands (standard credit-risk convention)
  iv_medium: 0.1
  iv_strong: 0.3
  woe_min_bin_frac: 0.05        # each WoE bin ≥5% of rows
  max_woe_bins: 10
correlation:
  redundant_abs: 0.95           # |corr| ≥0.95 → redundant pair
```

**Validation:** assert globs resolve, keys exist in at least one shard, referenced
entities exist, target join keys exist, `order` lists match data domains where
possible.

## 4. Discovery & cataloging

Per entity:

1. **Resolve shards** from `glob`; sort deterministically (natural sort so
   `2_RBIH` < `10_RBIH`).
2. **Read schema only** from each shard (parquet metadata — cheap). Record per-shard
   `num_rows`, byte size, column set, arrow types.
3. **Schema reconciliation across shards**: union of columns; flag columns
   missing from some shards, dtype disagreements, column-order differences.
   (Banking dumps often drift between extract batches — catch it here.)
4. **Memory estimate** → decide load strategy: full load vs `profile_row_cap`
   stratified sample (sampling config). Record which was used so every downstream
   number is interpretable.
5. Emit `catalog.json`: entity → {shards, total_rows, unified_schema, schema_drift,
   load_plan}.

**Output keys:** `n_shards, total_rows, unified_columns[], schema_conflicts[], load_plan`.

---

# Stage 2 — Analyze

Everything computed per column / per entity: typing, profiling, quality,
within-entity relationships, and drift. Heavy ops respect `heavy_op_sample`.

## 5. Semantic type inference (the brain of agnosticism)

Physical dtype (`large_string`, `int64`, `double`, `timestamp`) is **not enough** —
`PINCODE` is `int64` but is categorical-geographic; `ANNUAL_INCOME` is a *string*
holding a number; `*_FLAG` strings are booleans. So a **semantic_type** is assigned
that everything downstream branches on.

### 5.1 Resolution order
1. **Config override wins** (exact name, then glob pattern like `*_FLAG`).
2. Else **infer** from a sample using the ladder below.

### 5.2 Semantic type catalog & inference rules

| semantic_type   | Inferred when… | Example cols |
|-----------------|----------------|--------------|
| `boolean_flag`  | ≤2–3 distinct non-null in {Y/N,1/0,T/F,TRUE/FALSE} | `UPI_FLAG`, `PAN_FLAG` |
| `binary`        | exactly 2 distinct, not flag-coded | `CREDIT_DEBIT_INDICATOR` |
| `categorical`   | unique ≤ `categorical_max_unique` or ratio ≤ `categorical_max_ratio` | `ACCOUNT_TYPE`, `MODE_OF_TRANSACTION` |
| `ordinal`       | categorical **and** config gives `order` | `KYC_RATING`, `FRAUD_LAYER` |
| `categorical_id`| integer-coded category, not orderable-as-magnitude | `BRANCH_CODE` |
| `count`         | non-negative integers, name `NUM_*`, low-to-mid range | `NUM_CHEQUE_BOOKS` |
| `continuous`    | float, wide range, high uniqueness | `LIEN_AMOUNT` |
| `amount`        | continuous + monetary (config or name `*_AMOUNT/*_BALANCE`) | `TRANSACTION_AMOUNT` |
| `amount_string` | string that parses to number ≥ threshold | `ANNUAL_INCOME`, `TXN_AMOUNT` |
| `datetime`      | timestamp dtype, or string parses as date ≥ threshold | `ACCOUNT_OPEN_DATE`, `TRANSACTION_DATE` |
| `identifier`    | uniqueness ≥ `id_uniqueness_min` and is a key | `TRANSACTION_ID`, `ACCOUNT_NUMBER` |
| `hashed_pii`    | high-card hex/base64-looking, config `HASH_OF_*` | `HASH_OF_AADHAR` |
| `pincode`       | config / 6-digit integer geographic | `PINCODE`, `CUSTOMER_PIN_CODE` |
| `phone`         | 10–12 digit numeric, config | `MOBILE_NUMBER` |
| `ip_address`    | matches IPv4/IPv6 regex | `INITIATOR_IP_ADDRESS` |
| `email`         | matches email regex | — |
| `vpa`           | matches `user@psp` (UPI VPA) | `*_VIRTUAL_ADDRESS` |
| `bank_code`     | IFSC pattern `^[A-Z]{4}0[A-Z0-9]{6}$` | `*_IFSC` |
| `free_text`     | long strings, very high cardinality, whitespace/words | `ADDRESS`, `TRANSACTION_NARRATION`, `BENEFICIARY_NAME` |
| `constant`      | 1 distinct value | (any dead column) |
| `unknown`       | nothing matched → profiled generically, flagged for human | — |

### 5.3 Coercion plan
For each column record the **safe cast** to apply before profiling (and reusable by
FE): `amount_string → float (strip ₹, commas)`, `datetime string → datetime64`,
`boolean_flag → bool`, leave `identifier` as string. Record parse-failure rate; if a
cast fails for >X%, downgrade the type and flag it (no exception).

**Output keys:** `semantic_type, source(override|inferred), confidence, coercion, parse_fail_rate`.

## 6. Univariate profiling (core EDA, comprehensive)

Computed per column **conditioned on semantic_type**. Universal block runs for all;
type-specific blocks add to it.

### 6.1 Universal (every column)
- `count, n_missing, pct_missing, n_present`
- `n_unique, cardinality_ratio`
- `n_blank/whitespace`
- `top_values[ (value,count,pct) × top_k ]`
- `is_constant`, `is_near_constant` (top value ≥ dominance threshold)
- **entropy** (Shannon) + normalized entropy → diversity measure

### 6.2 Numeric (`continuous, amount, count, amount_string→numeric`)
- min, max, range, mean, **trimmed mean**, median
- std, variance, **MAD**, IQR, CV (std/mean)
- **skewness, kurtosis** → distribution shape tag (normal/right-skew/heavy-tail)
- percentiles: 1,5,10,25,50,75,90,95,99
- **outliers**: count + fraction by configured method (IQR/z/MAD)
- **histogram** (fixed bins) for the report
- sign profile (all-positive? → log-transform candidate)
- `is_integer_valued`, `zero_inflation` fraction (matters for amounts/counts → FE: zero-flag feature)

### 6.3 Categorical / ordinal / boolean / binary
- full or top-k **value frequency table**
- cardinality + **imbalance** (max class share, Gini, entropy)
- **rare categories** (< min frequency) count → FE: rare-bucketing
- for boolean_flag: true-rate, mapping used (Y/N→1/0)
- for ordinal: order coverage (missing/unexpected levels)
- **trailing/leading whitespace & case variants** of same label ("Upi" vs "UPI")
  → cleaning recommendation

### 6.4 Datetime
- parse success rate (post-coercion)
- min/max date, **span**, n_unique days
- **granularity** (has time component?) auto-detected
- distribution by year, day-of-week, hour (if time present)
- **future dates** count, **epoch/sentinel** dates (1900-01-01, 1970-01-01) flag
- derived-feature stubs: age-from-DOB, recency (max date − value), tenure

### 6.5 Identifier / hashed_pii
- uniqueness ratio, **duplicate count** (critical for keys)
- format consistency (length distribution, charset)
- For FK candidates: defer overlap to the relational pass. Do **not** treat as a
  model feature directly (flag `feature_role: key`).

### 6.6 Pattern-typed (`ip_address, email, vpa, bank_code, phone, pincode`)
- **validity rate** against the type's regex/checksum (IFSC format, IPv4 octets,
  pincode 6-digit, phone length)
- structural decomposition stubs → FE: IP→subnet, VPA→PSP handle (`@oksbi`),
  IFSC→bank+branch, email→domain, pincode→region/state (needs reference table → human flag)
- invalid-value examples (capped) for the report

### 6.7 Free text
- length stats (chars, tokens), language-agnostic
- emptiness, whitespace ratio, embedded-digit fraction
- → FE: text features are **stubbed and flagged** (vectorization is a later/optional
  stage; may need human/LLM — not done here).

**All heavy ops (entropy on huge cardinality, KDE) respect `heavy_op_sample`.**

## 7. Data-quality & sanity checks

Turns the profile into pass/warn/fail signals (your orchestrator's "sanity check"
gate). Per column:

- **Completeness**: missing% vs warn/drop thresholds → `OK | WARN | DROP_SUGGESTED`
- **Constancy**: constant / near-constant → `DROP_SUGGESTED`
- **Cardinality sanity**: high-card non-ID strings → likely free-text/ID mislabel
- **Type integrity**: parse-failure rate, mixed-type cells, numbers-as-strings
- **Range/sign**: negative amounts where impossible, future dates, sentinel dates
- **Duplicates**: full-row duplicates; key duplicates (PK must be unique)
- **Referential integrity** (set up here, measured in the relational pass): FK values present in parent
- **Whitespace/case** issues → standardize recommendation
- **Format validity** for pattern types (below configured rate → flag)

Per entity:
- **row-level dup rate**, PK uniqueness pass/fail
- **schema-drift across shards** (from the catalog) surfaced as WARN
- **quality score** = weighted roll-up → single number per entity for dashboards

**Output:** `quality_status, quality_flags[], quality_score`, and an **actions[]**
list of machine-readable remediation suggestions
(`{op: drop|impute|cast|trim|standardize_case|cap_outliers, reason, params}`)
that FE will consume. Nothing is auto-dropped silently; everything is recorded.

## 8. Multivariate / relationships (within entity)

On the typed, coerced (sampled) frame, **within each entity**:

- **Numeric ↔ numeric**: Pearson matrix → top correlated pairs;
  `|corr| ≥ redundant_abs` → **redundancy clusters** (FE: keep one per cluster).
- **Categorical ↔ categorical**: **Cramér's V** association matrix.
- **Functional dependencies**: A→B (B constant within each A) → e.g.
  `ACCOUNT_NUMBER → BRANCH_CODE`; informs which attributes are account-level vs txn-level.
- **Interaction candidates**: flag pairs with strong association for FE crosses
  (only *suggested*, built later).

**Output:** `top_numeric_corr[]`, `redundant_pairs[]`, `categorical_associations[]`,
`functional_dependencies[]`. (Persisted inside `profile.json` under `relationships`.)
## We are not able to split the differences between the various chunks of the data and not able to verify it with 
## and update the code within a suitable 
## 9. Temporal split suggestion

Shards are arbitrary size-splits of one population (not separate batches/time
periods), so cross-shard distribution comparison would be meaningless and is not
computed. If the entity has a time column, reserve a **time-based split point**
suggestion (train/valid) at its 80th percentile so FE and modeling can avoid
leakage. Recorded, not enforced.

**Output:** `suggested_time_split`.

---

# Stage 3 — Score

Target-aware ranking and cross-entity join characterization. Auto-skips (no error)
when no label resolves.

## 10. Target-aware analysis

**Gate:** if `config/target.yaml` missing, or all `positive_if_present_in` entities
missing/empty, set `supervised: false`, record `reason`, **skip — no error**.

When a target resolves:

1. **Attach label** per entity via `applies_to` (config join keys; bridge through a
   parent entity if `via:` given). Positives = membership in label entities (with
   optional `filter`); everyone else = 0 (`default_negative`). Record
   `positive_rate`, `n_pos`, `n_neg`, base rate.
2. Per feature column (excluding keys/PII):
   - **Numeric**: class-conditional separation; **KS statistic**,
     **point-biserial corr**, **single-feature AUC**.
   - **Categorical/boolean/pattern**: **Information Value (IV)** + **WoE** per bin
     (monotonic binning for numerics, `max_woe_bins`, `woe_min_bin_frac`);
     **Cramér's V** vs target; **target rate per category**.
3. **Rank** all columns by a combined predictive score (normalized IV / AUC /
   point-biserial), tag strength via IV bands (`weak/medium/strong`).
4. **Leakage guard**: flag columns that are near-perfectly predictive or are
   downstream of the label (e.g. anything sourced from `mha_*` joined back) →
   `leakage_suspect: true`, excluded from auto-FE priority.

Every active target is scored **separately** (so victim and fraudster signal stay
distinct), and the **primary target** drives feature-plan priority / report order.

**Output (per target):** `supervised(bool), positive_rate, iv, iv_band, woe_bins[],
auc, ks, point_biserial, target_rate_by_cat{}, predictive_rank, leakage_suspect`.

## 11. Cross-entity relational EDA

Uses config keys/foreign_keys to characterize **joins** (the basis for aggregate
features across `accounts`/`customers`/`transactions`):

- **Join cardinality**: classify each relationship N:1 / N:M (e.g.
  `transactions.ACCOUNT_NUMBER → accounts` is N:1; `accounts → transactions` fans 1:N).
- **Fan-out stats**: children per parent (txns per account): mean/median/p95/max →
  sizing for aggregation features.
- **Coverage / orphans**: % of FK values with no parent (orphan rate), % of parents
  with no children → data-integrity + FE coverage.
- **Aggregation map**: for each parent entity, list child numeric/categorical/datetime
  columns aggregatable to the parent grain (sum/mean/max/count/nunique/recency) →
  this is the **menu the FE step builds aggregate features from** (e.g. per-account:
  txn count, total/avg amount, distinct beneficiaries, days-since-last-txn,
  fraud-complaint count).

**Output:** `relationships[] {from,to,cardinality,orphan_rate,fanout_stats,
aggregation_candidates[]}`.

---

# Stage 4 — Output

## 12. Feature recommendation synthesis (the FE contract)

This is where EDA **becomes** automated feature engineering instructions. For every
column, deterministically map its profile → recommended FE operations. **No LLM, no
guessing data meaning** — pure rules over the metrics already computed. Operations
needing domain knowledge are emitted with `requires_human: true` and a clear prompt.

### 12.1 Rule examples (semantic_type → FE ops)

| Profile signal | Recommended FE op(s) |
|----------------|----------------------|
| `boolean_flag` | map→0/1; (impute missing→0 or "missing" flag) |
| `categorical`, low-card, has target signal | WoE-encode (from target bins) **or** one-hot; rare-bucket |
| `categorical`, high-card | target/frequency/hash encoding; rare-bucket; **flag** if grouping needs domain |
| `ordinal` | integer-encode by config `order` |
| `continuous`/`amount`, right-skew + all-positive | `log1p`; robust-scale; outlier cap at p99 |
| `amount` with zero-inflation | add `is_zero` flag + transform on non-zero |
| `count` | keep; add `has_any` flag; cap extreme |
| `datetime` | decompose (year/month/dow/hour); **recency** = ref−value; **tenure** |
| `DATE_OF_BIRTH` (derive_age) | age = ref−DOB; age bands |
| `pincode` | region/state lookup **(requires_human: reference table)**; frequency-encode |
| `ip_address` | subnet/prefix extraction; shared-IP count across accounts (cross-entity) |
| `vpa` | extract PSP handle (`@oksbi`); known-PSP flag |
| `bank_code` (IFSC) | extract bank + branch code; same-bank-as-account flag |
| `email`/`phone` | domain extract / validity flag; shared-contact count (cross-entity) |
| `free_text` | length, token-count, regex-entity flags now; **vectorization deferred** (requires_human/LLM) |
| `identifier`/`hashed_pii` | `feature_role: key` — not a model feature; used for joins/aggregation only |
| redundancy cluster | keep one representative; drop rest (suggested) |
| high missing | impute (median/mode/constant) + missingness flag, or drop if > drop threshold |
| cross-entity child cols | aggregate to parent grain (sum/mean/max/count/nunique/recency) |
| strong IV (target) | mark `priority: high`; ensure included; consider interactions |
| `leakage_suspect` | `exclude_from_model: true`, reason recorded |

### 12.2 Per-column FE record (emitted)

```json
{
  "entity": "transactions",
  "column": "TRANSACTION_AMOUNT",
  "semantic_type": "amount",
  "feature_role": "feature",
  "recommended_ops": [
    {"op": "transform", "fn": "log1p", "reason": "skew=4.2, all_positive"},
    {"op": "outlier_cap", "at": "p99", "reason": "outliers=0.154"},
    {"op": "scale", "method": "robust"}
  ],
  "target_signal": {"iv": 0.41, "iv_band": "strong", "predictive_rank": 3},
  "priority": "high",
  "requires_human": false,
  "leakage_suspect": false
}
```

Plus **cross-entity aggregate feature suggestions** (from the aggregation menu)
emitted at entity level, and a top-level **feature_plan.json** = the full
machine-readable contract the FE step loads. FE does **not** re-derive anything; it
just executes the plan (and human edits).

## 13. Artifacts & reporting

### 13.1 Machine-readable (the contract — FE consumes this)
```
artifacts/
  catalog.json                 # schemas, row counts, schema drift, load plan
  relational.json              # cross-entity joins + aggregate feature suggestions
  feature_plan.json            # THE contract (per-entity plan files + aggregates)
  <entity>/
    columns.parquet            # one row per column, all metrics (flat, queryable)
    profile.json               # full profile + quality rollup + within-entity relationships
    drift.json                 # suggested train/valid time split
    target.json                # per-target IV/AUC/KS + leakage flags (high-card capped, no bin tables)
    feature_plan.json          # per-column recommended FE ops (uniform schema)
```
`columns.parquet` is deliberately flat/tabular so downstream code (and humans in SQL)
can filter e.g. `iv_band == 'strong' AND requires_human == false`.

Per-column quality verdicts (status, flags, remediation actions) flow into
`feature_plan.json` and `columns.parquet`, so the only quality data unique to
`profile.json` is the entity-level rollup. The Score stage caps high-cardinality
categorical/pattern columns to the top-N values plus an `__other__` bucket before
computing IV, so IDs/geo (BRANCH_CODE, PINCODE, VPA, IP) get an honest IV instead of
per-value overfit, and the WoE bin tables are not persisted.

### 13.2 Human-viewable report (markdown)
```
report/
  <entity>.md                  # per-column table: type, missing, unique, quality,
                               #   IV per target (side-by-side), rank, priority, FE ops
```
Report is plain **markdown** (diff-able, PR-able, renders in any viewer) generated
from the same in-memory results that are persisted — single source of truth, **no
HTML, no notebook, no LLM**.

---

## 14. Graceful-degradation rules (so the orchestrator never breaks)

| Situation | Behavior |
|-----------|----------|
| Label config absent / label entity folder missing | `supervised=false`, skip scoring, continue |
| Entity marked `optional` and missing | skip entity, note in manifest |
| Column type can't be inferred | `semantic_type=unknown`, run universal profile, flag for human |
| Cast/parse fails for a column | downgrade type, record `parse_fail_rate`, never raise |
| Too few rows for a metric (e.g. WoE bin) | omit metric with `reason: insufficient_data` |
| Heavy op too costly | run on `heavy_op_sample`, record `sampled=true` |
| Schema drift across shards | WARN in quality, reconcile by column union |
| Anything FE needs that requires data meaning | emit op with `requires_human: true` + prompt |

---

## 15. Module layout

```
eda/
  config/ {datasource.yaml, target.yaml, thresholds.yaml}
  eda_engine/
    __init__.py
    orchestrator.py      # runs the 4 stages per entity; CLI entrypoint
    setup.py             # config load/validate + manifest; catalog & load plan
    analyze.py           # typing, univariate profiling, quality, relationships, drift
    score.py             # target-aware IV/WoE/AUC/KS + leakage; cross-entity joins
    output.py            # feature-plan synthesis; artifacts writer + markdown report
    utils/               # shared, data-agnostic helpers
      typing_registry.py # semantic types + inference rules
      metrics/           # pure stat fns (entropy, psi, iv, cramers_v, ...)
      io.py              # parquet/sample loaders, artifact read/write
  artifacts/   report/
  EDA_PROCEDURE.md       # this file
```

Key agnosticism guarantee: **`metrics/` and the stage modules import nothing about
PNB.** The only files that mention `ACCOUNT_NUMBER`, `mha_l0`, "fraud", etc. are
under `config/`.

---

## 16. End-to-end flow (one run)

```
Setup:   validate config ─► catalog all entities
Analyze: per entity → type ─► profile ─► quality ─► relationships ─► drift
Score:   per entity → target-aware rank (if label available) ─► cross-entity relational
Output:  synthesize feature_plan ─► write artifacts + report
```

FE step (later) = load `feature_plan.json` → execute non-`requires_human` ops
automatically → surface `requires_human` items to a person → (optionally) LLM-assist
those at the very end. EDA itself stays 100% deterministic and LLM-free.
