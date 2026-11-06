# accounts

- rows: **668,884**  ·  columns: **48**  ·  shards: **3**  ·  quality score: **85.8**
- mode: **supervised**  ·  targets: is_fraudster_account, is_victim_account
- primary target: `is_fraudster_account`  (positive_rate=0.1260)

> ⚠️ schema drift across shards: 1 issue(s)

| column | type | role | missing | unique | quality | IV[is_fraudster_account] | IV[is_victim_account] | rank | priority | FE ops |
|---|---|---|---|---|---|---|---|---|---|---|
| FREEZE_CODES | categorical | feature | 0.0% | 222 | OK | 2.949 (strong) | 0.8167 (strong) | 1 | high | encode, rare_bucket |
| LIEN_AMOUNT | amount | feature | 0.0% | 104593 | OK | 0.4213 (strong) | 2.195 (strong) | 2 | high | outlier_cap, add_flag, scale |
| ACCOUNT_OPEN_MODE | categorical | feature | 0.0% | 21 | OK | 0.281 (medium) | 0.3635 (strong) | 3 | medium | encode, rare_bucket |
| ACCOUNT_TYPE | categorical | feature | 0.0% | 175 | OK | 0.1666 (medium) | 0.2621 (medium) | 4 | medium | encode, rare_bucket |
| MPASSBOOK_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.05856 (weak) | 0.02474 (weak) | 5 | low | map_binary |
| NUM_CHEQUE_BOOKS | count | feature | 0.0% | 204 | OK | 0.04075 (weak) | 0.2304 (medium) | 6 | low | outlier_cap, add_flag, scale |
| KYC_RATING | categorical | feature | 0.0% | 3 | OK | 0.03921 (weak) | 0.05201 (weak) | 7 | low | encode |
| AREA_CATEGORY | boolean_flag | feature | 0.0% | 2 | OK | 0.03527 (weak) | 0.0356 (weak) | 8 | low | map_binary |
| SCHEME_TYPE | categorical | feature | 0.0% | 6 | OK | 0.02563 (weak) | 0.02314 (weak) | 9 | low | encode, rare_bucket |
| BRANCH_CODE | categorical_id | feature | 0.0% | 10180 | OK | 0.02004 (weak) | 0.07226 (weak) | 10 | low | encode, rare_bucket |
| PINCODE | pincode | feature | 0.0% | 5683 | OK | 0.01915 (useless) | 0.08626 (weak) | 11 | low |  **(human)** |
| VOTER_ID_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.01851 (useless) | 0.004486 (useless) | 12 | low | map_binary |
| NUM_CREDIT_CARDS | count | feature | 0.0% | 8 | OK | 0 (useless) | 0 (useless) | 13 | low | add_flag, scale |
| CHEQUE_BOOK_AVAILED_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.01491 (useless) | 0.1628 (medium) | 14 | low | map_binary |
| HAS_NOMINATION | boolean_flag | feature | 0.2% | 2 | OK | 0.0113 (useless) | 0.0116 (useless) | 15 | low | map_binary |
| PAN_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.01012 (useless) | 0.09015 (weak) | 16 | low | map_binary |
| ACCOUNT_STATUS | binary | feature | 0.7% | 2 | WARN | 0.0101 (useless) | 0.002505 (useless) | 17 | low | encode, rare_bucket |
| MOBILE_BANKING_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.008074 (useless) | 0.02193 (weak) | 18 | low | map_binary |
| BRANCH_CATEGORY | categorical | feature | 0.1% | 6 | OK | 0.006166 (useless) | 0.08322 (weak) | 19 | low | encode, rare_bucket |
| PMJJBY_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.005861 (useless) | 0.007514 (useless) | 20 | low | map_binary |
| CREDIT_CARD_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.005366 (useless) | 0.02187 (weak) | 21 | low | map_binary |
| SSA_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.004016 (useless) | 0.005918 (useless) | 22 | low | map_binary |
| PPF_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.002853 (useless) | 0.01964 (useless) | 23 | low | map_binary |
| LOCKER_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.002545 (useless) | 0.043 (weak) | 24 | low | map_binary |
| INTERNET_BANKING_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.001641 (useless) | 0.07638 (weak) | 25 | low | map_binary |
| SCSS_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.0007579 (useless) | 0.02286 (weak) | 26 | low | map_binary |
| CHEQUE_ALLOWED_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.0007312 (useless) | 0.1694 (medium) | 27 | low | map_binary |
| UPI_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.0006893 (useless) | 0.006557 (useless) | 28 | low | map_binary |
| UDHYAM_FLAG | boolean_flag | feature | 99.7% | 2 | DROP_SUGGESTED | 0.0006033 (useless) | 0.001331 (useless) | 29 | drop | drop, map_binary |
| GSTN_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.0003764 (useless) | 0.001981 (useless) | 30 | low | map_binary |
| PASSPORT_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 0.0003632 (useless) | 0.0009445 (useless) | 31 | low | map_binary |
| KYC_COMPLIANT_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.0002229 (useless) | 0.0005022 (useless) | 32 | low | map_binary |
| ATM_ISSUED_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.0001801 (useless) | 0.01491 (useless) | 33 | low | map_binary |
| PMJDY_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 0.0001065 (useless) | 0.1558 (medium) | 34 | low | map_binary |
| DEMAT_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 3.17e-05 (useless) | 0.008987 (useless) | 35 | low | map_binary |
| APY_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 2.363e-05 (useless) | 0.004822 (useless) | 36 | low | map_binary |
| AADHAR_FLAG | boolean_flag | feature | 0.0% | 2 | OK | 2.198e-05 (useless) | 0.008234 (useless) | 37 | low | map_binary |
| FAST_TAG_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 1.314e-05 (useless) | 0.002884 (useless) | 38 | low | map_binary |
| MUTUAL_FUND_FLAG | boolean_flag | feature | 0.0% | 2 | WARN | 3.597e-07 (useless) | 0.001527 (useless) | 39 | low | map_binary |
| ACCOUNT_NUMBER | identifier | key | 0.0% | 668884 | OK |  |  |  | unranked |  |
| CUSTOMER_ID | identifier | key | 0.0% | 665276 | OK |  |  |  | unranked |  |
| LAST_MOBILE_UPDATE_DATE | datetime | feature | 2.6% | 5318 | OK |  |  |  | unranked | datetime_decompose |
| CURRENCY_CODE | constant | feature | 0.0% | 1 | DROP_SUGGESTED |  |  |  | drop | drop |
| KYC_REVIEW_DATE | datetime | feature | 0.0% | 3854 | OK |  |  |  | unranked | datetime_decompose |
| LAST_FREEZE_DATE | datetime | feature | 44.2% | 318242 | WARN |  |  |  | unranked | impute, datetime_decompose |
| LAST_UNFREEZE_DATE | datetime | feature | 52.2% | 308078 | WARN |  |  |  | unranked | impute, datetime_decompose |
| ACCOUNT_OPEN_DATE | datetime | feature | 0.0% | 11857 | OK |  |  |  | unranked | datetime_decompose |
| LAST_TRANSACTION_DATE | datetime | feature | 0.0% | 976 | OK |  |  |  | unranked | datetime_decompose |

**Drop suggested (2):** `CURRENCY_CODE`, `UDHYAM_FLAG`
