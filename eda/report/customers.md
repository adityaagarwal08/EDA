# customers

- rows: **673,969**  ·  columns: **32**  ·  shards: **3**  ·  quality score: **78.8**
- mode: **supervised**  ·  targets: is_fraudster_account, is_victim_account
- primary target: `is_fraudster_account`  (positive_rate=0.1267)

> ⚠️ schema drift across shards: 4 issue(s)

| column | type | role | missing | unique | quality | IV[is_fraudster_account] | IV[is_victim_account] | rank | priority | FE ops |
|---|---|---|---|---|---|---|---|---|---|---|
| SAVING_ACCOUNT_BALANCE | amount | feature | 0.0% | 421684 | OK | 0.06688 (weak) | 0.18 (medium) | 1 | low | outlier_cap, scale |
| OCCUPATION | categorical | feature | 1.5% | 58 | OK | 0.03656 (weak) | 0.1114 (medium) | 2 | low | encode, rare_bucket |
| NUM_LOAN_ACCOUNT | count | feature | 0.0% | 18 | OK | 0 (useless) | 0 (useless) | 3 | low | outlier_cap, add_flag, scale |
| PERMANENT_PIN_CODE | pincode | feature | 2.3% | 25830 | OK | 0.02545 (weak) | 0.0689 (weak) | 4 | low |  **(human)** |
| CUSTOMER_PIN_CODE | pincode | feature | 0.1% | 25546 | OK | 0.02474 (weak) | 0.07063 (weak) | 5 | low |  **(human)** |
| NUM_SAVING_ACCOUNT | count | feature | 0.0% | 21 | OK | 0.001512 (useless) | 0.02998 (weak) | 6 | low | outlier_cap, scale |
| NUM_CREDIT_CARDS | count | feature | 0.0% | 5 | OK | 0 (useless) | 0 (useless) | 7 | low | add_flag, scale |
| LOAN_ACCOUNT_BALANCE | amount | feature | 0.0% | 24381 | OK | 0 (useless) | 0 (useless) | 8 | low | outlier_cap, add_flag, scale |
| NUM_TERM_DEPOSIT_ACCOUNT | count | feature | 0.0% | 93 | OK | 0 (useless) | 0 (useless) | 9 | low | outlier_cap, add_flag, scale |
| TERM_DEPOSIT_ACCOUNT_BALANCE | amount | feature | 0.0% | 17542 | OK | 0 (useless) | 0 (useless) | 10 | low | outlier_cap, add_flag, scale |
| NUM_CURRENT_ACCOUNT | count | feature | 0.0% | 18 | OK | 0 (useless) | 0 (useless) | 11 | low | outlier_cap, add_flag, scale |
| CREDIT_CARD_OUTSTANDING_AMOUNT | continuous | feature | 0.0% | 5242 | WARN | 0.003745 (useless) | 0.005569 (useless) | 12 | low | add_flag, scale |
| CURRENT_ACCOUNT_BALANCE | amount | feature | 0.0% | 12703 | OK | 0 (useless) | 0 (useless) | 13 | low | add_flag, scale |
| NUM_OVER_DRAFT_ACCOUNT | count | feature | 0.0% | 11 | WARN | 0 (useless) | 0 (useless) | 14 | low | add_flag, scale |
| GENDER | categorical | feature | 1.5% | 4 | OK | 0.008824 (useless) | 0.07776 (weak) | 15 | low | encode, rare_bucket |
| NATIONALITY | categorical | feature | 1.5% | 37 | WARN | 0.00715 (useless) | 0.002148 (useless) | 16 | low | encode, rare_bucket |
| MAX_SANCTIONED_LIMIT | count | feature | 0.0% | 173 | OK | 0 (useless) | 0 (useless) | 17 | low | add_flag, scale |
| KYC_VERIFIED_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.004909 (useless) | 0.006345 (useless) | 18 | low | map_binary |
| OVER_DRAFT_ACCOUNT_BALANCE | amount | feature | 0.0% | 2430 | WARN | 0 (useless) | 0 (useless) | 19 | low | add_flag, scale |
| CUSTOMER_SPECIAL_STATUS | categorical | feature | 6.9% | 8 | WARN | 0.001594 (useless) | 0.01359 (useless) | 20 | low | encode, rare_bucket |
| ANNUAL_INCOME | amount_string | feature | 1.8% | 17 | WARN |  |  | 21 | low | cast, scale |
| CUSTOMER_ID | identifier | key | 0.0% | 673969 | OK |  |  |  | unranked |  |
| DATE_OF_BIRTH | datetime | feature | 0.0% | 25496 | OK |  |  |  | unranked | datetime_decompose |
| HASH_OF_AADHAR | hashed_pii | key | 3.2% | 652055 | OK |  |  |  | unranked |  |
| HASH_OF_PAN | hashed_pii | key | 32.4% | 454977 | WARN |  |  |  | unranked |  |
| HASH_OF_VOTER_ID | hashed_pii | key | 96.1% | 26143 | DROP_SUGGESTED |  |  |  | drop |  |
| HASH_OF_PASSPORT | hashed_pii | key | 99.9% | 564 | DROP_SUGGESTED |  |  |  | drop |  |
| HASH_OF_DRIVING_LICENCE | hashed_pii | key | 99.8% | 1305 | DROP_SUGGESTED |  |  |  | drop |  |
| HASH_OF_NREGA | hashed_pii | key | 99.9% | 391 | DROP_SUGGESTED |  |  |  | drop |  |
| KYC_REVIEW_DATE | datetime | feature | 0.0% | 3865 | OK |  |  |  | unranked | datetime_decompose |
| CUSTOMER_OPEN_DATE | datetime | feature | 0.0% | 7497 | OK |  |  |  | unranked | datetime_decompose |
| ADDRESS | free_text | feature | 0.1% | 656769 | OK |  |  |  | unranked | text_basic, vectorize **(human)** |

**Drop suggested (4):** `HASH_OF_VOTER_ID`, `HASH_OF_PASSPORT`, `HASH_OF_DRIVING_LICENCE`, `HASH_OF_NREGA`
