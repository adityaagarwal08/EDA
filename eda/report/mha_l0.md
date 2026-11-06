# mha_l0

- rows: **119,843**  ·  columns: **4**  ·  shards: **1**  ·  quality score: **100.0**
- mode: **unsupervised**
- primary target: `is_fraudster_account`

| column | type | role | missing | unique | quality | rank | priority | FE ops |
|---|---|---|---|---|---|---|---|---|
| ACKNOWLEDGEMENT_NUMBER | count | feature | 0.0% | 119843 | OK |  | unranked | outlier_cap, scale |
| COMPLAINT_DATE | datetime | feature | 0.0% | 306 | OK |  | unranked | datetime_decompose |
| ACCOUNT_NUMBER | hashed_pii | key | 0.6% | 111710 | OK |  | unranked |  |
| DISPUTED_AMOUNT | amount | feature | 0.0% | 27060 | OK |  | unranked | outlier_cap, scale |
