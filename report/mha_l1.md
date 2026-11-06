# mha_l1

- rows: **208,572**  ·  columns: **7**  ·  shards: **1**  ·  quality score: **85.7**
- mode: **unsupervised**
- primary target: `is_fraudster_account`

| column | type | role | missing | unique | quality | rank | priority | FE ops |
|---|---|---|---|---|---|---|---|---|
| ACKNOWLEDGEMENT_NUMBER | count | feature | 0.0% | 106086 | OK |  | unranked | outlier_cap, scale |
| ACCOUNT_NUMBER | unknown | key | 0.0% | 92026 | OK |  | unranked |  |
| FRAUD_LAYER | ordinal | feature | 0.0% | 1 | DROP_SUGGESTED |  | drop | drop, ordinal_encode |
| TXN_ID | amount_string | feature | 0.0% | 196439 | OK |  | unranked | outlier_cap, scale |
| TXN_DATE | datetime | feature | 0.0% | 1030 | OK |  | unranked | datetime_decompose |
| TXN_AMOUNT | amount_string | feature | 0.0% | 16738 | OK |  | unranked | outlier_cap, transform, scale |
| DISPUTED_AMOUNT | amount_string | feature | 0.0% | 16491 | OK |  | unranked | outlier_cap, scale |

**Drop suggested (1):** `FRAUD_LAYER`
