# transactions

- rows: **128,048**  ·  columns: **26**  ·  shards: **10**  ·  quality score: **86.2**
- mode: **supervised**  ·  targets: is_fraudster_account, is_victim_account
- primary target: `is_fraudster_account`  (positive_rate=0.1231)

| column | type | role | missing | unique | quality | IV[is_fraudster_account] | IV[is_victim_account] | rank | priority | FE ops |
|---|---|---|---|---|---|---|---|---|---|---|
| MOBILE_NUMBER | phone | feature | 5.1% | 25048 | OK | 1.449 (strong) | 2.442 (strong) | 1 | high | extract |
| BENEFICIARY_VIRTUAL_ADDRESS | vpa | feature | 9.1% | 31690 | OK | 1.434 (strong) | 1.467 (strong) | 2 | high | extract, extract |
| REMITTER_VIRTUAL_ADDRESS | vpa | feature | 8.6% | 29705 | OK | 1.287 (strong) | 1.721 (strong) | 3 | high | extract, extract |
| REMITTER_IFSC | bank_code | feature | 6.2% | 11361 | OK | 0.721 (strong) | 2.154 (strong) | 4 | high | extract, extract |
| INITIATOR_DEVICE_OS | categorical | feature | 9.0% | 324 | OK | 0.2622 (medium) | 0.4198 (strong) | 5 | medium | standardize_case_trim, encode, rare_bucket |
| TRANSACTION_INITIATOR_DETAILS | categorical | feature | 47.5% | 7 | WARN | 0.1723 (medium) | 0.1478 (medium) | 6 | medium | impute, encode, rare_bucket |
| TRANSACTION_ID | amount_string | feature | 1.9% | 125256 | OK | 0.1426 (medium) | 0.06291 (weak) | 7 | medium | outlier_cap, transform, scale |
| MODE_OF_TRANSACTION | categorical | feature | 0.0% | 8 | OK | 0.1417 (medium) | 0.8679 (strong) | 8 | medium | encode, rare_bucket |
| TRANSACTION_AMOUNT | amount | feature | 0.0% | 6864 | OK | 0.0481 (weak) | 0.2891 (medium) | 9 | low | outlier_cap, transform, scale |
| BENEFICIARY_BANK | categorical | feature | 94.0% | 66 | DROP_SUGGESTED | 0.09103 (weak) | 0.5704 (strong) | 10 | drop | drop, standardize_case_trim, encode, rare_bucket |
| REMITTER_BANK | categorical | feature | 93.8% | 87 | DROP_SUGGESTED | 0.09065 (weak) | 0.8002 (strong) | 11 | drop | drop, standardize_case_trim, encode, rare_bucket |
| TRANSACTION_TYPE | categorical | feature | 44.6% | 13 | WARN | 0.07318 (weak) | 0.2711 (medium) | 12 | low | impute, encode, rare_bucket |
| CREDIT_DEBIT_INDICATOR | binary | feature | 1.9% | 2 | OK | 0.04209 (weak) | 0.2661 (medium) | 13 | low | encode |
| INITIATOR_IP_ADDRESS | ip_address | feature | 10.9% | 24271 | OK | 0.03416 (weak) | 0.03636 (weak) | 14 | low | extract, extract |
| BENEFICIARY_IFSC | bank_code | feature | 3.0% | 9729 | OK | 0.03396 (weak) | 1.138 (strong) | 15 | low | extract, extract |
| TRANSACTION_SUB_TYPE | categorical | feature | 50.5% | 3 | WARN | 0.0242 (weak) | 0.1134 (medium) | 16 | low | impute, encode, rare_bucket |
| ACCOUNT_NUMBER | categorical | key | 0.0% | 200 | OK |  |  |  | unranked | encode, rare_bucket |
| BENEFICIARY_NAME | free_text | feature | 2.7% | 24375 | OK |  |  |  | unranked | text_basic, vectorize **(human)** |
| BENEFICIARY_ACCOUNT_NUMBER | unknown | feature | 2.3% | 19992 | OK |  |  |  | unranked |  |
| TRANSACTION_DATE | datetime | feature | 0.0% | 1205 | OK |  |  |  | unranked | datetime_decompose |
| TRANSACTION_NARRATION | free_text | feature | 5.2% | 120476 | OK |  |  |  | unranked | text_basic, vectorize **(human)** |
| REMITTER_NAME | free_text | feature | 2.0% | 21921 | OK |  |  |  | unranked | text_basic, vectorize **(human)** |
| REMITTER_ACCOUNT_NUMBER | unknown | feature | 0.0% | 27394 | OK |  |  |  | unranked |  |
| INITIATOR_APPLICATION_NAME | free_text | feature | 5.1% | 131 | OK |  |  |  | unranked | text_basic, vectorize **(human)** |
| TRANSACTION_TIME | datetime | feature | 0.2% | 57279 | OK |  |  |  | unranked | datetime_decompose |
| CUSTOMER_ID | categorical | key | 35.7% | 2867 | WARN |  |  |  | unranked | impute, encode, rare_bucket |

**Drop suggested (2):** `BENEFICIARY_BANK`, `REMITTER_BANK`
