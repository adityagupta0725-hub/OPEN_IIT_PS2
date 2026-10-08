# PS2 Project Handoff: Full Context of the Chat So Far

## 0. Read this first (instructions for the AI model receiving this document)

This document is a complete handoff of a working session between a user and Claude (Anthropic's assistant). The user is a participant in a competition run by **CreditNirvana** (AI for debt collections in India) and is working on **Problem Statement 2 (PS2): Right-Party Contact Prediction and Skip-Trace Prioritisation**.

- The user has some machine learning background and is strong in maths and standard data structures and algorithms.
- **No code has been written or run yet.** Everything so far is analysis, planning, data profiling and specification. The numbers quoted in the data sections come from profiling the CSV files with pandas.
- The document has three parts: **Part A** (this section: context, conversation timeline, problem, data dictionary, findings, decisions, status), **Appendix B** (the README: architecture, repo layout, models and maths), and **Appendix C** (the PIPELINE: a stage-by-stage build spec with acceptance checks).
- Please use Part A as shared memory, then continue from "Section 8: Status and next steps". When writing code, follow Appendix C stage by stage.
- Things that were inferred rather than verified are marked "inferred" or "unverified". Do not treat them as facts.

---

## 1. Conversation timeline (what the user asked, what was done)

| # | User request | What was done |
|---|---|---|
| 1 | Shared a Google Doc link with the competition problem statements. Asked for a solution approach to PS2, different approaches, why each step, alternatives, optimisations, and a brief list of PS-specific points | The link was private (HTTP 401). The user's connected Google Drive had a recently shared file **CN_Problem_Statements_v1.docx**, whose PS2 is "Right-Party Contact Prediction and Skip-Trace Prioritisation". It was read and used. **Assumption: this is the same document as the user's link. The user should confirm.** A full approach was written: a three-layer design (state estimator, action policy, trace valuation) with alternatives at each step |
| 2 | Asked for full forms of HMM and GBDT, and the models and maths used at each implementation step | Explained HMM (Hidden Markov Model) and GBDT (Gradient Boosted Decision Trees), listed all acronyms, and gave the maths for each step (labelling, GBDT updates, forward algorithm, Beta decay, hazard, CUSUM, calibration, SHAP, EVT, Thompson sampling, IPS/DR, shrinkage) |
| 3 | Asked for a README.md compiling the structure, models and maths needed to implement | Created `README.md` (reproduced in Appendix B) |
| 4 | Uploaded the 11 competition CSV files. Asked what each file is, how to use it, why columns are missing, and what each column means | Profiled every file (shape, nulls, value counts, joins, relationships). Produced a file-by-file data dictionary, an explanation of null patterns, and a list of information that is absent. Found important data traps (see Section 5) |
| 5 | Asked for a procedural pipeline from cleaning to modelling to tuning so any chatbot can write the code | Created `PIPELINE.md` (reproduced in Appendix C): 20 stages with inputs, outputs, formulas and acceptance checks |
| 6 | Asked for a PDF and then a Markdown file of the whole chat context to send to a friend with another AI model | This document |

---

## 2. The problem (PS2), in brief

**Task:** For every phone number and address on file, predict the probability that it reaches the borrower (a **right-party contact**, RPC). Then decide per account whether to keep trying, switch contact point, switch channel, trace, or suppress. Skip-traces are ranked by the **expected recovery value** of finding the borrower, not by a fixed attempt count.

**Requirements and constraints (from the problem statement):**
- The unit of prediction is the **contact point** (a phone or an address), not the account.
- "**Avoiding**" (a live person who will not engage) and "**invalid/dead**" must be separated, because the right actions differ (switch channel vs trace).
- True labels are never fully observed. Only contact points that were tried and logged correctly produce outcomes.
- **Recycled numbers** (a phone reassigned to a new person) are a compliance risk: calling them can disclose the debt to a stranger. The target is **zero disclosure incidents**, so errors are asymmetric: favour recall on these classes.
- Contact-point health **changes over time**, so the model must decay or track state (the problem statement itself suggests a survival or hidden-state model).
- The trace decision is a **value-of-information** calculation.
- The model's own dial order shapes its future data (a **feedback loop**), so a small explicit exploration budget is needed.
- Scores must be **refreshed daily**, **explainable** (reason codes), and respect a zero-disclosure **compliance layer**.
- Out of scope: in-call identity verification.
- Phasing: phones first (weeks 1 to 6), addresses and field evidence next (weeks 7 to 12).
- Success metrics: RPC per 1,000 dials, AUC and calibration per contact-point type, attempts wasted on invalid points, trace hit rate and recovery, third-party disclosures (zero).
- Data note from the statement: data and sandbox access "can be provided". The user has since uploaded the 11 CSV files.

**Core design idea:** split the work into three layers: (1) a **state estimator** per contact point, (2) an **action policy** per account, (3) a **trace valuation** per account. Each can be built, tested and defended separately.

---

## 3. Dataset overview

Period covered by call logs: **2026-04-01 to 2026-06-29**. Ground-truth labels were collected on **2026-07-02** (after the logs end, so no leakage).

```
accounts (2,400 borrowers) -- splits (train/validation/test per account)
   |
   |-- phones (5,719 rows) ---- dial_attempts (51,105 calls) ---- agents
   |        '-- verified_contact_points (250 ground-truth labels)
   |
   |-- addresses (3,117) ------ field_visits (5,578 visits) ----- agents
   |
   |-- skip_traces (766) --> creates new phones / addresses
   |-- payments (2,162) --> the recovery outcome
   '-- lenders (6)
```

| File | Rows | What it is |
|---|---|---|
| accounts.csv | 2,400 | One row per borrower account: loan, delinquency, behaviour context |
| phones.csv | 5,719 | Inventory of phone contact points (no outcomes) |
| dial_attempts.csv | 51,105 | Every call attempt: the main event table |
| verified_contact_points.csv | 250 | Ground-truth status of selected phones |
| addresses.csv | 3,117 | Inventory of addresses |
| field_visits.csv | 5,578 | Every physical field visit with outcome |
| skip_traces.csv | 766 | Past skip-trace runs and results |
| payments.csv | 2,162 | Payments received (recovery outcome) |
| agents.csv | 30 | Callers and field agents (20 tele, 9 field, 1 bot) |
| lenders.csv | 6 | Lender metadata |
| splits.csv | 2,400 | Train 1,680 / validation 360 / test 360, by account |

---

## 4. Data dictionary (column meanings are inferred from names and values; no official dictionary was supplied)

### accounts.csv

| Column | Meaning | Use |
|---|---|---|
| account_id | Key | Join key |
| lender_id | Lender that owns the loan | Lender-level differences |
| portfolio | pl_salaried, two_wheeler, mfi_jlg, credit_card, consumer_durable, msme | Behaviour differs by product |
| income_type | salaried / self_employed / daily_wage | Reachability hours differ |
| town_id | T1 / T2 / T3 | Town priors, agent teams |
| preferred_language | hinglish / kanglish / english | Language match with agents |
| bucket_start | Delinquency bucket (1-30, 31-60, 61-90, 90+, X = not yet overdue) | Deep buckets raise avoidance |
| dpd_start | Days past due | Numeric delinquency |
| emi_amount | Monthly instalment | Debt scale |
| overdue_start | Overdue amount | Recovery value input |
| outstanding | Total outstanding | Key input to the trace value |
| salary_credit_day | Day of month salary arrives | Best contact timing |
| bureau_score_band | Credit score band | Risk proxy |
| other_active_loans | Number of other loans | Multi-loan stress; likely reason for shared phone IDs |
| paid_other_lenders_30d | Paid another lender recently | Strong avoidance signal (has money, not paying us) |
| last_bounce_reason | Last EMI bounce reason | Intent vs inability |
| ability_to_pay_estimate | 0 to 1 model score | Recovery value feature |
| prev_ptp_count, prev_ptp_broken | Past promises and broken promises | Reliability |
| dialling_arm | rule_based or random_contact_point | Experiment arm |

### phones.csv

| Column | Meaning | Use |
|---|---|---|
| phone_id | Phone identifier (**not unique**) | Join to attempts |
| account_id | Borrower | Link |
| source | kyc_origination, reference, borrower_update, bureau, employer, skip_trace | Source quality prior |
| relation_recorded | self, reference_spouse, office, etc. | Reference numbers are more likely third parties |
| added_date | When the number was added | Age of the contact point |
| phone_masked | Last 4 digits only | **Do not use as identity** |
| priority_slot | Current dial order (0 = first) | Baseline ranking to beat |

### dial_attempts.csv

| Column | Meaning | Use |
|---|---|---|
| attempt_id, account_id, phone_id | Keys | Joins |
| attempt_ts | Call timestamp | Time-slot features, recency, decay |
| channel | tele_agent or voice_bot | Human vs bot behaviour |
| agent_id | Caller | Agent effect, noisy logging |
| dialling_arm | rule_based or random_contact_point | Experiment arm |
| selection_propensity | Probability this phone was chosen | Needed for IPS/DR evaluation |
| network_response | answered, ring_no_answer, busy_rejected, not_reachable, switched_off, number_does_not_exist | Telecom-level outcome; the cleanest signal |
| ring_duration_s | Ringing time | Human-present vs dead number |
| talk_duration_s | Conversation length | Contact quality |
| hangup_by | network / customer / agent | Customer hangup suggests a live person |
| disposition | Agent or bot outcome code (16 values) | Label source (noisy) |
| remark | Free text (Hinglish / Kannada-English) | Text features |
| ptp_id | Promise-to-pay ID, only when a promise was made | Success marker |
| has_transcript | Whether a transcript exists (~7%) | Transcript text itself is NOT provided |

Disposition groups: **RPC** = rpc_ptp, rpc_hung_up, rpc_call_back, rpc_refused, rpc_hardship, rpc_dispute, rpc_claims_paid. **Third party** = third_party_contact, third_party_ptp. **Bad number** = wrong_number, invalid_number. **No contact** = no_answer, call_rejected, not_reachable, switched_off. **Other** = language_barrier.

### verified_contact_points.csv (ground truth)

| Column | Meaning | Use |
|---|---|---|
| phone_id, account_id | Keys | Join |
| verified_status | borrower_number (127), third_party_number (74), not_borrower_number (27), switched_off (19), invalid_number (3) | The only real ground truth: use for evaluation and calibration, not bulk training |
| verified_date | All 2026-07-02 | Labels collected after the logs end |

### addresses.csv

| Column | Meaning | Use |
|---|---|---|
| address_id, account_id | Keys | Join |
| address_type | residence / office / permanent_native | Where the borrower likely is |
| source | kyc_origination (3,090) or skip_trace (27) | Skip-trace addresses are fresher |
| added_date | When added | Age |
| town_id | T1/T2/T3 or OUT | OUT means outside operating towns (mostly permanent_native) |
| address_text | Free text, multilingual (including Kannada script) | Parsing, standardisation (PS3-adjacent) |

### field_visits.csv

| Column | Meaning | Use |
|---|---|---|
| visit_id, account_id, address_id, agent_id | Keys | Join |
| visit_date, start_ts, checkin_ts | Timing | Gap between start and check-in = travel/search effort |
| checkin_x, checkin_y | Check-in position (looks like local metres, about +/-4,400) | Was the agent really at the address? |
| gps_accuracy_m | GPS error radius | Trust weight |
| dwell_s | Time at the address | met_borrower median ~890s; address_not_traceable ~77s |
| outcome | met_borrower, met_family, locked_premises, neighbour_says_shifted, no_such_person, address_not_traceable, cash_collected | Address label source |
| ptp_id | Promise made, if any | Success marker |
| remark | Free text | Text features |
| photo_hash | Hash of the visit photo | 161 duplicates: possible fraud signal (reused photos) |

### skip_traces.csv

| Column | Meaning | Use |
|---|---|---|
| trace_id, account_id, trace_date | Keys, timing | Join |
| trigger_rule | Always 15_consecutive_failed_contacts | **The current baseline rule to beat** |
| result | no_new_info (591), new_phone_found (148), new_address_found (27) | Trace success label (hit rate ~23%) |
| new_contact_point_id | ID of the new phone or address found | Links to phones/addresses (all matched) |
| cost_inr | Trace cost (INR 60 to 150) | C_trace in the value formula |

### payments.csv

| Column | Meaning | Use |
|---|---|---|
| payment_id, account_id | Keys | Join |
| payment_ts | When paid | Outcome timing |
| amount | Amount paid | Recovery value |
| channel | upi_link, upi_app, nach_represent, branch_cash, neft, bbps, cash_field | Collection route; cash_field ties to field visits |

Note: some payments (nach_represent, branch_cash) happen with no contact at all, so contact-driven recovery must be separated from self-cure.

### agents.csv, lenders.csv, splits.csv

| File | Columns | Meaning |
|---|---|---|
| agents.csv | agent_id, channel, language_team, town_id, tenure_months, shift | TA = tele, FA = field, BOT01 = bot. Day shift only, so no night calls exist |
| lenders.csv | lender_id, lender_name, lender_type, kyc_address_format | kyc_address_format (kyc_mixed, kyc_upper, app_free_text) matters for address parsing |
| splits.csv | account_id, split | Use as given: split is by account, so no borrower leaks across folds |

---

## 5. Key data findings and traps (verified by profiling)

1. **phone_id is not unique.** 175 rows in phones.csv share an id with another row, and the same phone_id appears under different account_ids (likely the same physical number on several accounts). Use `(account_id, phone_id)` as the per-account key, and `phone_id` alone for cross-account graph features.
2. **phone_masked holds only the last 4 digits**, and 1,066 masked values map to several different phone_ids. It must never be used as an identity key.
3. **"Answered" does not mean right party.** In the 250 verified phones, third_party_number has the highest answer rate (about 47%), higher than genuine borrower_number (about 28%). This is the heart of the problem.
4. **The experiment arm is the only unbiased data.** About 95% of calls come from the rule-based dialler (selection_propensity = 1.0, same numbers always picked). Only about 5% of accounts (123) are in random_contact_point, with propensities 0.5, 0.3333, 0.25 (one over the number of candidate phones). Training only on the rule-based data would copy the old dialler's bias.
5. **A large share of phones is unlabeled.** About half of reference / bureau / employer phones were never dialled. 32 accounts have zero attempts. These are unlabeled, not negative (a PU-learning situation).
6. **Dispositions are noisy.** Network response and disposition are consistent, but agent-entered dispositions can disagree with call metadata (for example rpc_hung_up has a median talk time of 216s).
7. **Calls are only 08:00 to 18:59** (heavy 08 to 12). Agents are day shift only.
8. **Verified labels leak nothing** (collected 2026-07-02) but are tiny: 173 / 41 / 36 verified phones sit in train / validation / test accounts. Use them as a gold evaluation and calibration set.
9. **Possible fraud / quality signal:** reused photo_hash values in field visits; very short dwell times.
10. **The trace baseline hit rate is about 23%** (148 phones plus 27 addresses found out of 766 traces), trigger fixed at 15 consecutive failed contacts.

---

## 6. Missing values: why they exist

### A. Empty cells (nulls). Mostly structural, so treat the null as information

| Where | Null rate | Probable reason (inferred) | What to do |
|---|---|---|---|
| dial_attempts.ptp_id | 94% | Only calls that ended in a promise get an id | Make a made_ptp flag |
| field_visits.ptp_id | 89% | Same | Same |
| skip_traces.new_contact_point_id | 77% | Empty exactly when result = no_new_info | Not a data problem |
| agents.town_id | 70% | Tele agents and the bot are not tied to a town | Fill with "none" |
| agents.tenure_months | 1 row | Probably the bot | Leave null |
| accounts.salary_credit_day | 68% | Probably not applicable to non-salaried borrowers (unverified; salaried is only 53% of accounts) | Keep a has_salary_day flag; do not impute |
| accounts.ability_to_pay_estimate | 30% | Scoring model likely produced no score (unverified) | Keep null plus a missing flag; trees handle nulls |

Do not mean-impute these.

### B. Information absent from the dataset

| Absent | Consequence |
|---|---|
| Per-phone state label (valid / avoiding / recycled ...) | Labels must be built; only 250 phones have verified truth |
| Explicit RPC flag | Derive from disposition (noisy) |
| Transcript text (only a has_transcript flag) | Only remark text is usable |
| PTP table (ptp_id is a dangling reference) | Promise amount, date and kept/broken are not visible; infer from payments |
| Telecom metadata (operator, SIM age, HLR status, reassignment date) | Recycled-number detection must be inferred from behaviour |
| Address coordinates (only visit check-in x/y) | Cannot verify check-in against address location directly (PS3-related) |
| True state for never-tested phones | Unlabeled; needs PU learning or exploration |
| Dial cost | Only trace cost is given |
| Real phone numbers (masked) | Identity must come from phone_id |

---

## 7. Design decisions made so far

1. **Three layers:** state estimator, action policy, trace valuation.
2. **Two prediction tasks.** Task 1: per-attempt next-call outcome (P(rpc), P(answered)), plentiful labels, used for dial ranking. Task 2: per-phone state as of 2026-06-29, with weak labels from rules, evaluated against the 250 verified phones.
3. **Hierarchical state model:** first "is a live human present?" (binary GBDT), then "which state?" (multiclass GBDT), combined by multiplying probabilities.
4. **Time dynamics built in order:** Beta-Bernoulli with exponential decay (baseline), then discrete-time hazard model, then an input-output HMM (hidden-state tracking with a forward filter).
5. **Recycled / third-party detection:** CUSUM change detection plus a character n-gram text classifier plus graph signals, combined by noisy-OR; threshold chosen for very high recall (target 99%) against gold labels.
6. **Cross-channel evidence** (engagement on other phones, payments, paid_other_lenders_30d) is the key to distinguishing avoiding from invalid.
7. **Weak supervision:** labelling functions, soft labels, PU correction (Elkan-Noto) for never-tested phones. Gold labels are never used as training features and are used only for evaluation, calibration and tuning on train-split gold.
8. **Calibration:** Platt or isotonic per target on validation / out-of-fold predictions; report ECE and Brier.
9. **Decision layer:** compliance overrides first (suppress), then a rule table over calibrated probabilities; reason codes from TreeSHAP.
10. **Trace value:** `EVT = P(find) x P(recover | reached) x Outstanding x uplift - C_trace`; rank by EVT per rupee under a budget; compare against the 15-consecutive-failures baseline.
11. **Exploration and evaluation:** Thompson sampling with a 2-5% budget; log propensities; evaluate with IPS / doubly robust using the random-arm accounts. Expect wide confidence intervals because there are only about 123 random-arm accounts.
12. **Evaluation protocol:** use splits.csv as given; test set touched once; point-in-time features (a leakage unit test is mandatory); tune with grouped cross-validation by account within train.
13. **Addresses (phase 2):** same architecture, with outcomes mapped to address states and hierarchical shrinkage toward pincode or town.
14. **Tools:** Python 3.11, pandas, numpy, scipy, scikit-learn, LightGBM, Optuna, SHAP, networkx, hmmlearn (optional), pyarrow.

---

## 8. Status and next steps

**Done:** problem analysis, approach design, maths, data profiling and dictionary, README (Appendix B), pipeline spec (Appendix C).

**Not done:** no code has been written or run; no labels built; no models trained; no results exist.

**Placeholders / to verify:**
- Weak-label rule thresholds (for example 0.5 RPC share, 0.7 switched-off share) are placeholders and must be tuned against the verified phones (train-split gold only for tuning).
- The address-text pincode regex (6 digits at the end of the string) is based on a few sample addresses and must be checked first.
- The exact telecom number-reallocation window (commonly cited as about 90 days in India) should be verified before being hardcoded.
- The column meanings in Section 4 are inferred.
- The user's original Google Doc link was not accessible; the problem statement used is CN_Problem_Statements_v1.docx from the user's Drive. Confirm it is the right PS2.

**Suggested next step:** implement stages S1 to S4 of Appendix C (load and validate, clean, keys, labels), then check how well the simple weak-label rules agree with the 250 verified phones. Then build the baselines (S10) before any complex model.

**Open questions for the organisers** (also in Appendix B): the exact RPC definition; how never-tested contact points are represented; cost values for dials and visits; whether any telecom signals (HLR, reallocation data) are available.

---

# Appendix B: README.md (architecture, repo layout, models and maths)

# PS2: Right-Party Contact (RPC) Prediction and Skip-Trace Prioritisation

> Source: CreditNirvana Problem Statement 2 (AI-native debt collections, India).
> This README is the implementation blueprint: architecture, repo layout, models, and the maths each module must implement.

---

## 1. Problem summary

For every **phone number** and **address** on file, predict the probability that it reaches the borrower (a **right-party contact**, RPC). A decision layer then chooses, per account, one of:

`continue` | `switch contact point` | `switch channel` | `skip-trace` | `suppress`

Skip-traces are ranked by the **expected recovery value** of finding the borrower, not by a fixed attempt count.

### Hard requirements (from the problem statement)

| Requirement | Implication for design |
|---|---|
| Unit of prediction is the **contact point**, not the account | Event table keyed by `(contact_point_id, attempt)` |
| "Avoiding" vs "invalid" must be separated | Evidence-of-live-human features + hierarchical classifier |
| True labels are never observed | Weak supervision + PU learning, soft labels |
| Health changes over time | Decay / hazard / HMM state tracking |
| Recycled numbers risk debt disclosure | Asymmetric threshold (high recall), suppression list, **0 disclosure target** |
| Trace is a value-of-information decision | EVT formula + budgeted greedy selection |
| Feedback loop from own dial order | Exploration budget + logged propensities |
| Scores refreshed **daily** | Batch pipeline, incremental features |
| Explainable and auditable | Reason codes via SHAP, hard rule layer |
| Out of scope | In-call identity verification |

### Phasing

- **Weeks 1-6:** phone numbers (health score, avoiding-vs-invalid, recycled check, dialer and suppression integration).
- **Weeks 7-12:** addresses and field evidence, PS3 location confidence, account-level trace decision across all contact points.

---

## 2. Acronyms

| Acronym | Meaning |
|---|---|
| RPC | Right-Party Contact |
| HMM | Hidden Markov Model |
| GBDT | Gradient Boosted Decision Trees (LightGBM / XGBoost) |
| PU | Positive-Unlabeled (learning) |
| SHAP | SHapley Additive exPlanations |
| AUC | Area Under the ROC Curve |
| ECE | Expected Calibration Error |
| CUSUM | Cumulative Sum change detection |
| IPS | Inverse Propensity Scoring |
| DR | Doubly Robust (estimator) |
| EVT | Expected Value of Tracing |
| EM | Expectation-Maximization |

---

## 3. System architecture

```
Raw logs (dialer, dispositions, remarks, bot transcripts, field app, account records)
        |
        v
[1] Ingestion + event table  ---->  [2] Label engine (weak supervision, PU)
        |                                   |
        v                                   v
[3] Feature store (per contact point, per day) <-- [5] Shared-contact graph
        |
        v
[4] State estimator
      4a. Human-present GBDT          \
      4b. State GBDT (multiclass)      }--> calibrated state posteriors
      4c. Time dynamics (decay/hazard/HMM) /
      4d. Recycled-number detector (CUSUM + text + graph) --> suppression list
        |
        v
[6] Calibration (per contact-point type / portfolio)
        |
        v
[7] Decision policy (rule table over posteriors + compliance overrides)
        |                         |
        v                         v
[8] Trace valuation (EVT)    [9] Exploration (Thompson sampling, propensity logging)
        |
        v
[10] Outputs: dial list order, action + reason code, trace queue, suppression list, dashboard
        |
        v
[11] Evaluation (temporal split, IPS/DR, safety recall, business metrics)
```

---

## 4. Proposed repository structure

```
ps2-rpc/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- default.yaml            # decay lambda, thresholds, budgets, costs
|   `-- states.yaml             # state taxonomy + allowed transitions
|-- data/
|   |-- raw/                    # real data if provided
|   `-- synthetic/              # simulator output
|-- src/
|   |-- simulate/
|   |   `-- simulator.py        # hidden-state generator for contact points
|   |-- ingest/
|   |   `-- events.py           # build (contact_point, attempt) event table
|   |-- labels/
|   |   |-- labeling_functions.py
|   |   |-- label_model.py      # weak supervision combiner
|   |   `-- pu.py               # Elkan-Noto correction
|   |-- features/
|   |   |-- telephony.py
|   |   |-- text.py             # Hinglish remark/transcript features
|   |   |-- graph.py            # shared-contact graph
|   |   `-- build.py
|   |-- models/
|   |   |-- baseline_beta.py    # Beta-Bernoulli with decay
|   |   |-- gbdt_human.py       # stage A
|   |   |-- gbdt_state.py       # stage B
|   |   |-- hazard.py           # discrete-time survival
|   |   |-- hmm.py              # forward algorithm, covariate transitions
|   |   |-- recycled.py         # CUSUM + text + noisy-OR
|   |   `-- calibrate.py        # Platt / isotonic
|   |-- decision/
|   |   |-- policy.py           # action rule table + compliance overrides
|   |   |-- trace_value.py      # EVT + greedy budgeted selection
|   |   `-- explore.py          # Thompson sampling + propensity logging
|   |-- explain/
|   |   `-- reason_codes.py     # SHAP -> plain-language reasons
|   |-- eval/
|   |   |-- metrics.py          # AUC, Brier, ECE, RPC per 1k dials
|   |   |-- ope.py              # IPS / DR off-policy evaluation
|   |   `-- backtest.py         # wasted-attempts simulation vs fixed rule
|   `-- pipeline/
|       `-- daily_run.py        # end-to-end daily batch
|-- notebooks/
|   `-- 01_eda.ipynb ... 05_results.ipynb
`-- tests/
```

---

## 5. Data model

### 5.1 Event table (one row per attempt)

| Column | Description |
|---|---|
| `contact_point_id` | Phone or address ID |
| `borrower_id`, `account_id`, `lender_id` | Keys |
| `cp_type` | `phone` / `address` |
| `ts` | Attempt timestamp (hour slot derived) |
| `channel` | `tele` / `field` / `whatsapp` / `sms` / `bot` |
| `ring_duration`, `answered`, `call_duration` | Telephony |
| `network_response` | `ok` / `switched_off` / `not_reachable` / `does_not_exist` |
| `disposition` | e.g. `RPC`, `wrong_number`, `third_party` |
| `remark_text` | Agent remark (Hinglish) |
| `transcript_text` | Voice-bot transcript |
| `rpc_flag` | Logged right-party contact |
| `outcome_payment` | Paid / PTP after contact |

### 5.2 Contact-point table (static)

`source` (KYC, bureau, borrower-provided, update), `age_days`, `ever_confirmed_by_payment`.

### 5.3 State taxonomy

Phone: `valid_reachable`, `valid_avoiding`, `temp_unreachable`, `off_long_term`, `recycled`, `third_party`, `invalid`.
Address: `valid_occupied`, `valid_absent_usually`, `moved`, `hard_to_find`, `fabricated_incomplete`.

---

## 6. Modules, models and maths

### 6.1 Synthetic simulator (`simulate/`)

Needed because real data may be limited and for ground-truth testing.

- Sample a hidden state sequence per contact point from a transition matrix `A`.
- Emit attempt logs from state-conditional emission distributions (answer rate by hour slot, ring duration, network response, hangup speed).
- Inject: shared numbers across borrowers, recycled events (long-off gap followed by wrong-number dispositions), untested contact points, and noisy agent logging (a fraction of RPCs unlogged).
- Output both observations and the true state, so every later layer can be scored against truth.

### 6.2 Label engine (`labels/`)

**Labelling functions** (rules) map evidence to a state or abstain:

| Evidence | Label |
|---|---|
| RPC logged and payment or PTP followed | `valid_reachable` |
| Network `does_not_exist` | `invalid` |
| Long run of `switched_off` | `off_long_term` |
| "wrong number" / "galat number" disposition or remark | `recycled` or `third_party` |
| Answered by relative | `third_party` |
| Never tested | **unlabeled** (never treated as negative) |

**Label model (naive Bayes combiner):**

```
log[ P(y=1 | lambda) / P(y=0 | lambda) ] = prior_logodds + sum_k w_k * lambda_k
w_k = log( acc_k / (1 - acc_k) )
```

Rule accuracies `acc_k` are estimated without ground truth (Snorkel-style).

**PU correction (Elkan-Noto)** for partially labelled positives (`s=1` means labelled):

```
P(y=1 | x) = P(s=1 | x) / c,     c = P(s=1 | y=1)   (estimated on held-out known positives)
```

**Soft-label training loss:**

```
L = - sum_k q_k * log p_k           (q = soft label vector, p = model softmax)
```

### 6.3 Feature engineering (`features/`)

Telephony features:
- `answer_rate_slot[h] = answers_h / attempts_h` for each hour slot h
- `hangup_fast_rate` = share of answered calls shorter than ~3s
- `ring_duration_mean`, `ring_duration_std`
- `days_since_last_RPC`, `days_since_last_answer`
- `consecutive_unreachable_count`
- network response counts by type (last 7/30/90 days)

**Cross-channel evidence (key for avoiding vs invalid):**
- `engaged_other_channel_30d` (payment-link open, WhatsApp read, answered another number)

Text features (Hinglish): character n-gram TF-IDF (n = 2..5) over remarks and transcripts. Character n-grams absorb spelling variation ("band hai", "bnd hai").

Record features: `source`, `age_days`, `ever_confirmed_by_payment`.

### 6.4 Shared-contact graph (`features/graph.py`)

Bipartite graph: borrowers on one side, contact points on the other.

- Degree: `d(c)` = distinct borrowers linked to contact point c. Use `log(1 + d(c))`.
- Connected components via union-find (near-linear time).
- IDF-style down-weighting: `w(c) = 1 / log(1 + d(c))`.
- Fraction of links where the number is a reference or co-borrower, not a primary contact.

A number tied to many unrelated borrowers suggests a reference, shared, or recycled contact point.

### 6.5 State estimator (`models/`)

#### 6.5.1 Stage A: human present? (binary GBDT)

GBDT update:

```
F_0 = initial prediction
for m = 1..M:
    r_i = - dL/dF (F_{m-1}(x_i))            # residual / negative gradient
    fit tree h_m(x) to r
    F_m(x) = F_{m-1}(x) + eta * h_m(x)       # eta ~ 0.05
p = sigmoid(F(x)) = 1 / (1 + exp(-F(x)))
Loss: log-loss = -[ y log p + (1-y) log(1-p) ]
```

Key evidence: ring then fast hangup, declined calls, silent pickups, cross-channel engagement imply a human is present (avoiding). No ring or network `does_not_exist` implies no human (invalid).

#### 6.5.2 Stage B: which state? (multiclass GBDT)

```
p_k = exp(F_k) / sum_j exp(F_j)       # softmax over K states
```

Hierarchical combination:

```
P(state = k) = P(human present) * P(k | human present)     for human states
P(state = k) = (1 - P(human present)) * P(k | no human)    for non-human states
```

#### 6.5.3 Time dynamics (build in this order)

**(A) Beta-Bernoulli with exponential decay (baseline)**

```
weight_i = exp( -lambda * (t_now - t_i) )
alpha = alpha_0 + sum_i weight_i * s_i
beta  = beta_0  + sum_i weight_i * (1 - s_i)
P(reachable) = alpha / (alpha + beta)
Var = alpha*beta / ( (alpha+beta)^2 * (alpha+beta+1) )
```

Confidence falls automatically as evidence ages. Tune `lambda` on validation data.

**(B) Discrete-time hazard / survival**

```
h(t | x) = P( contact point fails at t | survived to t-1, x )
S(t) = prod_{u<=t} ( 1 - h(u | x_u) )
```

Fit `h` with a GBDT or logistic regression on person-period rows (one row per contact point per day). `S(t)` is the probability the contact point is still valid at time t.

**(C) Input-output HMM**

- Hidden states: the phone state taxonomy.
- Covariate-dependent transitions:

```
A_ij(x_t) = exp( w_ij . x_t ) / sum_k exp( w_ik . x_t )
```

- Irregular gaps between attempts (continuous time): `A(dt) = expm( Q * dt )`, where Q is a rate matrix (`scipy.linalg.expm`).
- Forward filtering (live belief over states):

```
alpha_t(j)  proportional to  b_j(o_t) * sum_i alpha_{t-1}(i) * A_ij
normalise so  sum_j alpha_t(j) = 1
```

- Emission `b_j(o)`: product of independent terms (answered, ring bucket, network response, hour slot), or a GBDT output converted to a likelihood.
- Training: Baum-Welch (EM) if unsupervised, or maximum likelihood using soft labels from 6.2 if semi-supervised.
- Output: filtered posterior `alpha_t(.)`, which feeds the decision layer.

**Recommended path:** ship (A), then (B) + (C) together. GBDT predicts per-attempt RPC probability given time slot and history, and the HMM tracks state with decay.

#### 6.5.4 Recycled-number detector (`models/recycled.py`)

Signature: history of RPCs, then a quiet or off period, then a shift to wrong-number dispositions or "I don't know this person". Telecom operators reallocate inactive numbers after a quiet period (commonly cited as about 90 days in India; verify before hardcoding).

**CUSUM change detection:**

```
S_t = max( 0, S_{t-1} + log[ p1(x_t) / p0(x_t) ] )
alarm when S_t > h
```

`p0` = outcome likelihood under "same subscriber", `p1` = under "new subscriber".

**Text classifier:** character n-gram TF-IDF with logistic regression, detecting Hinglish phrases such as "galat number", "yahan koi nahi rehta", "yeh kisi aur ka number hai".

**Graph signal:** number shared by many unrelated borrowers.

**Combine detectors (noisy-OR):**

```
P(recycled) = 1 - prod_i ( 1 - p_i )
```

**Asymmetric decision:** choose the action minimising expected loss.

```
a* = argmin_a  sum_s P(s) * L(a, s)
L(dial, recycled) >> L(suppress, valid)
```

In practice, pick threshold `tau` on the validation ROC so that **recall >= 99%** and report the false-positive rate it costs. Suppressed numbers can be re-verified through a safe channel.

#### 6.5.5 Calibration (`models/calibrate.py`)

```
Platt:     p_cal = sigmoid( a * score + b )
Isotonic:  fit monotone non-decreasing f minimising sum_i ( y_i - f(s_i) )^2   (PAV algorithm)
```

Calibrate per contact-point type and per portfolio. Validate on a **temporal split grouped by borrower** (never a random split).

```
Brier = (1/n) * sum_i ( p_i - y_i )^2
ECE   = sum_m ( |B_m| / n ) * | acc(B_m) - conf(B_m) |      (probability bins B_m)
AUC   = P( score of random positive > score of random negative )
```

### 6.6 Decision policy (`decision/policy.py`)

Rule table over calibrated posteriors, with a **hard compliance override layer applied first**.

| Condition | Action |
|---|---|
| State is `recycled` or `third_party` with high posterior | **Suppress.** No debt discussion |
| Best phone is valid and reachable | Dial at best time slot |
| Best phone is avoiding | Switch channel (WhatsApp, field) |
| Temporarily unreachable | Retry with exponential backoff |
| Off long-term or all points dead/invalid | Switch contact point, or trace |
| Address: valid and occupied | Visit |
| Address: valid but usually absent | Change visit time |
| Address: hard to find | Resolve location first (PS3), do not write off |
| Address: fabricated or moved | Trace, flag origination if fabricated |

Every action carries a **reason code**.

### 6.7 Reason codes (`explain/reason_codes.py`)

TreeSHAP gives each feature's contribution (Shapley value):

```
phi_i = sum_{S subset of F \ {i}}  [ |S|! * (M - |S| - 1)! / M! ] * [ f(S U {i}) - f(S) ]
```

Map the top 2-3 features to templated phrases, for example: "Answers 9-11am, hangs up within 2s, so likely avoiding. Recommend WhatsApp."

### 6.8 Skip-trace valuation (`decision/trace_value.py`)

```
EVT = P(trace finds valid CP) * P(recover | reached) * Outstanding * uplift - C_trace
```

- `uplift` = recovery gain over not tracing.
- `P(trace finds valid CP)`: separate GBDT trained on historical trace outcomes.
- Trace only if `EVT > 0`.
- Under a budget B, sort by `EVT / C_trace` and select greedily until B is spent (greedy fractional-knapsack approximation).
- Only trace accounts that are findable but not currently reachable.

### 6.9 Exploration (`decision/explore.py`)

Reserve about 2-5% of attempts for exploration.

- **Thompson sampling:** for each candidate contact point with posterior `Beta(alpha, beta)`, draw `theta ~ Beta(alpha, beta)` and rank by `theta`.
- Never explore suppressed recycled or third-party numbers. Compliance beats exploration.
- **Log propensity** `mu(a | x)` for every action taken.

### 6.10 Address module (phase 2)

Same architecture, different features:
- "locked premises" vs "nobody of that name here" outcomes
- GPS dwell at check-in, visit-time patterns
- PS3 location confidence as an input feature

Field data is sparse, so use **hierarchical shrinkage** toward the pincode or locality group g:

```
p_hat_addr = ( successes + kappa * mu_g ) / ( visits + kappa )
```

`mu_g` is the group mean success rate. `kappa` is the prior strength (pseudo-visits), fitted by **empirical Bayes**, i.e. maximum likelihood of the Beta-Binomial marginal across addresses.

---

## 7. Evaluation (`eval/`)

### 7.1 Protocol

- **Temporal split, grouped by borrower.** Train on earlier periods, validate and test on later ones. All contact points of a borrower stay in the same fold.
- Report per contact-point type (phone / address) and per portfolio.

### 7.2 Metrics

| Metric | Definition |
|---|---|
| AUC, Brier, ECE | As in 6.5.5 |
| RPC per 1,000 dial attempts, per 100 visits | Compared with the current rules |
| Wasted attempts on invalid contact points before action | Backtest vs fixed attempt-count rule |
| Skip-trace hit rate, recovery from traced accounts | Compared with fixed-count rule |
| Recycled-number recall at fixed FPR | Safety metric |
| Third-party disclosure incidents | Target: zero |

### 7.3 Off-policy evaluation (using exploration logs)

```
V_IPS = (1/n) * sum_i r_i * pi(a_i | x_i) / mu(a_i | x_i)

V_DR  = (1/n) * sum_i [ q_hat(x_i, pi(x_i))
                       + ( pi(a_i | x_i) / mu(a_i | x_i) ) * ( r_i - q_hat(x_i, a_i) ) ]
```

`q_hat` is a learned reward model. Clip importance weights (for example at 10-20) to control variance.

### 7.4 Ablation story to present

Beta-decay baseline, then plus GBDT, then plus hazard, then plus HMM, then plus graph and cross-channel features, then plus EVT trace ranking. Show measured lift at each step.

---

## 8. Compliance and safety design

- **Hard rule layer** sits above all models and cannot be overridden by scores.
- Recycled and third-party numbers go to a **suppression list**. The debt is never discussed with the wrong person (RBI Fair Practices Code).
- Bureau and reference data is used only for its collected purpose (DPDP Act).
- Skip-trace sources must be lawful and auditable.
- Every action has a reason code and a logged score, so it is auditable.
- Scores refresh **daily** before outbound planning.

---

## 9. Build order

1. Event schema and synthetic simulator.
2. Labelling functions, label model, PU correction.
3. Features (telephony, text, graph).
4. Beta-decay baseline plus calibration plus evaluation harness.
5. Stage A and Stage B GBDTs (avoiding vs invalid).
6. Hazard model, then HMM state tracking.
7. Recycled-number detector and suppression list.
8. Decision policy and reason codes.
9. EVT trace ranking and budgeted selection.
10. Exploration and IPS/DR evaluation.
11. Address module (phase 2).
12. Dashboard and demo.

---

## 10. Libraries

| Need | Library |
|---|---|
| GBDT | `lightgbm` (or `xgboost`) |
| HMM | `hmmlearn` for basics. Custom NumPy forward algorithm for covariate transitions |
| Survival | `lifelines`, or LightGBM on person-period rows |
| Calibration | `sklearn.isotonic`, `sklearn.calibration` |
| Explanations | `shap` |
| Graph | `networkx` or custom union-find |
| Text | `scikit-learn` (TF-IDF, `LogisticRegression`) |
| Matrix exponential | `scipy.linalg.expm` |
| Data | `pandas`, `numpy`, `pyarrow` |
| Config and tests | `pyyaml`, `pytest` |

`requirements.txt` starter:

```
numpy
pandas
scipy
scikit-learn
lightgbm
shap
networkx
hmmlearn
lifelines
pyyaml
pytest
matplotlib
```

---

## 11. Open questions to confirm with organisers

- Exact **RPC definition** (does an answer by the borrower with no payment intent count?).
- How **never-tested** contact points are represented in the data.
- Cost values: `C_trace`, cost per dial, cost per field visit.
- Whether any **telecom signals** (HLR lookup, operator reallocation data) are available.
- Whether real data or sandbox access will be provided, and the format.


---

# Appendix C: PIPELINE.md (stage-by-stage build specification)

# PS2 Build Pipeline: Contact-Point Health, RPC Prediction and Skip-Trace Prioritisation

A stage-by-stage specification. Each stage states **inputs, steps, outputs, and acceptance checks** so that any coding assistant (or person) can implement it without seeing earlier conversations.

---

## 0. How to use this document with a chatbot

1. Paste **Section 1 (context)** and **Section 2 (conventions)** at the start of every chat. They are the shared memory.
2. Then paste **one stage at a time** (Sections 3.S0, 3.S1, ...) with this wrapper:

```
You are a senior data scientist. Implement ONLY the stage below as a Python module
under src/ following the conventions in Section 2. Read inputs from the paths given,
write outputs to the paths given, and add assertions for every acceptance check.
Return: (1) the full code, (2) a short test (pytest), (3) how to run it.
Do not change the contracts of earlier stages. If something is ambiguous, state your
assumption in a comment and continue. [PASTE STAGE HERE]
```

3. After each stage runs, paste the printed summary back so the next stage can build on real numbers.
4. Never skip the acceptance checks. Most bugs in this problem are **leakage** or **wrong joins**, not model choice.

---

## 1. Problem and data context (paste this first)

**Task (PS2):** For every phone number and address on file, estimate the probability that contacting it reaches the borrower (right-party contact, RPC), classify its health state, decide the next action per account, and rank accounts for skip-trace by expected recovery value. Pre-call prediction only. Phones first, addresses second.

**Hard requirements:**
- Unit of prediction is the **contact point**, not the account.
- "Avoiding" (a live person who won't engage) and "invalid/dead" must be separated, because they need opposite actions.
- Recycled numbers and third-party numbers must be **suppressed** (calling them risks disclosing the debt). Prefer high recall on these classes, accept lower precision.
- Never-tested contact points are **unlabeled, not negative**.
- Probabilities must be **calibrated**.
- Every action needs a **reason code**.
- Existing skip-trace baseline rule: trigger after 15 consecutive failed contacts.

**Data files (all in `data/raw/`):** period 2026-04-01 to 2026-06-29. Gold labels collected 2026-07-02.

| File | Rows | Key columns |
|---|---|---|
| accounts.csv | 2,400 | account_id, lender_id, portfolio, income_type, town_id, preferred_language, bucket_start, dpd_start, emi_amount, overdue_start, outstanding, salary_credit_day, bureau_score_band, other_active_loans, paid_other_lenders_30d, last_bounce_reason, ability_to_pay_estimate, prev_ptp_count, prev_ptp_broken, dialling_arm |
| phones.csv | 5,719 | phone_id, account_id, source, relation_recorded, added_date, phone_masked, priority_slot |
| dial_attempts.csv | 51,105 | attempt_id, account_id, phone_id, attempt_ts, channel, agent_id, dialling_arm, selection_propensity, network_response, ring_duration_s, talk_duration_s, hangup_by, disposition, remark, ptp_id, has_transcript |
| verified_contact_points.csv | 250 | phone_id, account_id, verified_status, verified_date |
| addresses.csv | 3,117 | address_id, account_id, address_type, source, added_date, town_id, address_text |
| field_visits.csv | 5,578 | visit_id, account_id, address_id, agent_id, visit_date, start_ts, checkin_ts, checkin_x, checkin_y, gps_accuracy_m, dwell_s, outcome, ptp_id, remark, photo_hash |
| skip_traces.csv | 766 | trace_id, account_id, trace_date, trigger_rule, result, new_contact_point_id, cost_inr |
| payments.csv | 2,162 | payment_id, account_id, payment_ts, amount, channel |
| agents.csv | 30 | agent_id, channel, language_team, town_id, tenure_months, shift |
| lenders.csv | 6 | lender_id, lender_name, lender_type, kyc_address_format |
| splits.csv | 2,400 | account_id, split (train 1,680 / validation 360 / test 360) |

**Verified facts about the data (already checked):**
- `phone_id` is NOT unique in phones.csv (175 rows share an id; same id appears under different accounts). Use the pair `(account_id, phone_id)` as the per-account contact-point key, and `phone_id` alone for cross-account graph features.
- `phone_masked` holds only the last 4 digits, and 1,066 masked values map to multiple phone_ids. **Never use it as an identity key.**
- `network_response` values: answered, ring_no_answer, busy_rejected, not_reachable, switched_off, number_does_not_exist.
- `disposition` values (16): no_answer, call_rejected, not_reachable, switched_off, invalid_number, wrong_number, language_barrier, third_party_contact, third_party_ptp, rpc_ptp, rpc_hung_up, rpc_call_back, rpc_refused, rpc_hardship, rpc_dispute, rpc_claims_paid. Network response and disposition are consistent. Dispositions are agent-entered and noisy (e.g. `rpc_hung_up` has median talk time 216s).
- Verified statuses: borrower_number 127, third_party_number 74, not_borrower_number 27, switched_off 19, invalid_number 3. Third-party numbers have a HIGHER answer rate (about 47%) than genuine borrower numbers (about 28%), so "answered" does NOT mean right party.
- 173 / 41 / 36 verified phones fall in train / validation / test accounts respectively.
- `dialling_arm`: 95% of calls are `rule_based` (selection_propensity = 1.0). About 5% of accounts (123) use `random_contact_point` with propensities 0.5 / 0.3333 / 0.25 (one over number of candidate phones). That arm is the only unbiased sample of untried numbers.
- Roughly half of reference / bureau / employer phones were never dialled (unlabeled). 32 accounts have zero attempts.
- Calls happen only 08:00 to 18:59 (heavy 08 to 12, light 13 to 18). Agents are day shift only.
- Structural nulls: dial_attempts.ptp_id (94%), field_visits.ptp_id (89%), skip_traces.new_contact_point_id (77%, equals no_new_info), agents.town_id (70%, tele agents and bot), accounts.salary_credit_day (68%), accounts.ability_to_pay_estimate (30%).
- Not available: transcript text, a PTP table, telecom metadata (operator, SIM age, HLR), address coordinates, dial cost, per-phone state labels.
- Trace baseline hit rate is about 23% (148 phones + 27 addresses found out of 766). Trace cost is INR 60 to 150.

---

## 2. Conventions (paste this second)

- Python 3.11, `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `optuna`, `shap`, `networkx`, `pyarrow`, `pyyaml`, `matplotlib`, `pytest`. Optional: `hmmlearn`.
- Random seed `42` everywhere, set from `configs/default.yaml`.
- Store intermediate tables as **parquet** in `data/interim/` and `data/processed/`. Final scores in `outputs/`.
- Repo layout:

```
ps2/
  configs/default.yaml
  data/raw/  data/interim/  data/processed/
  src/
    config.py  io.py
    s1_validate.py  s2_clean.py  s3_keys.py  s4_labels.py  s5_combine.py
    s6_eda.py  s7_features.py  s8_graph.py  s9_text.py
    s10_baselines.py  s11_models.py  s12_recycled.py  s13_calibrate.py
    s14_tune.py  s15_decision.py  s16_ope.py  s17_address.py
    s18_evaluate.py  s19_daily_run.py
  tests/
  notebooks/
  outputs/
```

- **Split rule:** always use `splits.csv` as given. Never create a random row split. All rows of an account stay in one split.
- **Test set is touched once**, in S18 only. Tuning uses train (grouped CV) and validation.
- **Point-in-time rule (critical):** any feature for a prediction at time `t` may use only events with timestamp `< t`. Every feature function takes an `as_of` argument. A leakage unit test is mandatory (see S20).
- Every module exposes `run(config) -> None`, writes its outputs, and prints a summary (row counts, null rates, class balance).

`configs/default.yaml` should hold at least:

```yaml
seed: 42
paths: {raw: data/raw, interim: data/interim, processed: data/processed, out: outputs}
data_end: "2026-06-29"
label_date: "2026-07-02"
decay_lambda_days: 0.05          # tuned in S14
recent_windows_days: [7, 14, 30, 60]
trace_baseline_consecutive_fails: 15
recycled_target_recall: 0.99
explore_budget_share: 0.03
cost: {trace_default_inr: 102}
```

---

## 3. Stages

### S0. Setup
**Do:** create the repo layout, `config.py` (loads yaml), `io.py` (read/write parquet helpers), and a `Makefile` or `run_all.py` that executes S1 to S19 in order.
**Accept:** `python run_all.py --stage s1` runs and logs.

---

### S1. Load and validate
**Input:** all CSVs in `data/raw/`. **Output:** `data/interim/raw_validated/*.parquet` and `outputs/s1_validation_report.md`.

**Steps:**
1. Load each file with explicit dtypes. Parse timestamps (`attempt_ts`, `payment_ts`, `start_ts`, `checkin_ts`) and dates (`added_date`, `visit_date`, `trace_date`, `verified_date`).
2. Schema checks: expected columns present, no unexpected extras.
3. Uniqueness: primary keys unique (attempt_id, visit_id, trace_id, payment_id, address_id, account_id in accounts and splits). **phone_id is allowed to repeat**, report how many.
4. Referential integrity:
   - Every `dial_attempts.account_id` exists in accounts.
   - Every `(account_id, phone_id)` in dial_attempts exists in phones. Report mismatches (do not drop silently).
   - Every `field_visits.address_id` exists in addresses and belongs to the same account.
   - Every `skip_traces.new_contact_point_id` exists in phones or addresses.
   - Every verified `(account_id, phone_id)` exists in phones.
5. Range and sanity: dates within 2026-04-01 to 2026-06-29 (verified on 2026-07-02), `ring_duration_s` between 0 and 45, `outstanding` and `amount` positive.
6. Report null rates per column and confirm they match the structural-null list in Section 1.

**Accept:** a markdown report listing every check as PASS/WARN with counts. No silent row drops.

---

### S2. Cleaning
**Input:** validated tables. **Output:** `data/interim/clean/*.parquet`.

**Steps (per file):**
- **accounts:** keep `bucket_start` as categorical with order `X < 1-30 < 31-60 < 61-90 < 90+`. Add `ability_missing = ability_to_pay_estimate.isna()` and `has_salary_day = salary_credit_day.notna()`. Do not mean-impute. Cast bools.
- **phones:** drop exact duplicate rows only. Keep all rows that differ in `account_id`. Do NOT use `phone_masked` for any join.
- **dial_attempts:** add `hour = attempt_ts.hour`, `dow`, `date`. Normalise `remark` text (lowercase, strip, collapse whitespace, keep original in `remark_raw`). Flag impossible rows (e.g. `talk_duration_s > 0` with `network_response != answered`) into a `data_issue` column without dropping.
- **field_visits:** compute `travel_gap_s = checkin_ts - start_ts`, flag `photo_reused = photo_hash.duplicated(keep=False)`. Keep coordinates as given (units unknown, treat as local metres).
- **addresses:** extract `pincode` with regex `(\d{6})\s*$` (verify on a sample first), `text_len`, `has_kannada_script` (Unicode block U+0C80 to U+0CFF), `script_mix_ratio`, `has_landmark_words` (near, opp, behind, hatra, hathira, hinde, munde, etc.). Keep `address_text` raw.
- **skip_traces:** derive `found_new = result != 'no_new_info'`.
- **payments:** no changes beyond type casting.
- **agents:** fill agents.town_id null with `"none"`.

**Accept:** row counts unchanged except documented exact-duplicate removal. A `clean_summary.json` with before/after counts.

---

### S3. Keys and entity resolution
**Input:** clean tables. **Output:** `data/interim/contact_points.parquet`, `phone_account_links.parquet`.

**Steps:**
1. Build `cp_id = account_id + "|" + phone_id` for phones, and `address_id` for addresses.
2. Build the contact-point master table: one row per `cp_id` with source, relation_recorded, added_date, priority_slot, and `n_accounts_sharing_phone_id` (count of distinct account_ids for the same phone_id).
3. Build `phone_account_links` (phone_id, account_id) used later by the graph stage.
4. Attach `cp_id` to every dial attempt via `(account_id, phone_id)`.
5. Record never-tried contact points: `n_attempts == 0`.

**Accept:** `cp_id` is unique. Every attempt maps to exactly one `cp_id`. Report how many contact points have zero attempts, by source (expected: about half of reference / bureau / employer).

---

### S4. Label construction
Most important stage. Two label sets are built: a **per-attempt outcome label** (plentiful) and a **per-phone state label** (weak, evaluated on 250 verified).

**Input:** clean attempts, contact points, verified table. **Output:** `data/processed/attempt_labels.parquet`, `data/processed/phone_state_weak.parquet`, `data/processed/phone_state_gold.parquet`.

#### S4a. Attempt-level labels
Create columns on each attempt:
- `y_answered = network_response == 'answered'`
- `y_rpc = disposition.startswith('rpc_')`
- `y_third_party = disposition in {third_party_contact, third_party_ptp}`
- `y_wrong = disposition in {wrong_number, invalid_number}`
- `y_ptp = ptp_id.notna() & disposition in {rpc_ptp}` (and a separate `y_any_ptp`)
- `y_dead_signal = network_response in {number_does_not_exist}`
- `y_off_signal = network_response in {switched_off, not_reachable}`
- `y_human_present = hangup_by == 'customer' or network_response in {answered, busy_rejected}` (a live person rejected or answered; tune after EDA)

#### S4b. Phone-level weak state label (as of `data_end`)
States: `valid_reachable`, `valid_avoiding`, `temp_unreachable`, `off_long_term`, `third_party`, `recycled_or_wrong`, `invalid`, `unknown` (never tested or too little evidence).

Implement as **labelling functions**. Each returns a state or ABSTAIN, and the combiner is a majority vote with confidences. Starting rules (thresholds are placeholders, tune them against verified labels in S4d):

| Rule | State |
|---|---|
| any `number_does_not_exist` | invalid |
| at least 1 `rpc_*` and `rpc` share of answered calls above 0.5 | valid_reachable |
| answered calls mostly `third_party_*` or recorded relation is reference/office, and `rpc_*` share is low | third_party |
| at least 1 `rpc_*` earlier AND at least 2 `wrong_number` (or remark with "kisi aur ka", "naya number", "galat number") later | recycled_or_wrong |
| share of `switched_off`+`not_reachable` over last K attempts is at least 0.7 and no answer in the last 30 days | off_long_term |
| mostly `ring_no_answer`/`busy_rejected`/`call_rejected` AND (some human-present signal OR other contact point of the same account answered or the account has a payment in the window) | valid_avoiding |
| few attempts (under 3) | unknown |
| otherwise with mixed outcomes | temp_unreachable |

Output a **soft label** vector (probabilities over states) per phone by counting rule votes weighted by rule precision.

#### S4c. Gold labels
Map `verified_status` to model states:

| verified_status | State |
|---|---|
| borrower_number | valid_reachable or valid_avoiding (collapse to `borrower` for evaluation) |
| third_party_number | third_party |
| not_borrower_number | recycled_or_wrong |
| switched_off | off_long_term |
| invalid_number | invalid |

Keep gold labels **out of training features and weak-label rules**. They are for evaluation and calibration only.

#### S4d. Rule audit
Compare weak labels to gold for the 250 verified phones. Report a confusion matrix and per-rule precision/recall. Adjust thresholds using **train-split verified only** (173), check on validation (41), never on test (36).

**Accept:** label distribution table, confusion matrix vs gold, and confirmation that gold labels were not used inside rules for validation/test accounts.

---

### S5. Combine into modelling tables
**Output:** three tables in `data/processed/`.

1. **`attempt_table`** (one row per attempt): attempt fields + `cp_id` + labels + account context (join accounts) + agent attributes (join agents) + lender (join lenders) + `split`.
2. **`cp_snapshot_table`** (one row per contact point per `as_of` date): features computed point-in-time (S7), weak label, gold label if any, split. Build snapshots at weekly dates (e.g. every Monday) plus the final `data_end`.
3. **`account_table`** (one row per account): account features, counts of contact points, best-contact-point stats, trace history, payments, split.

**Joins to get right:**
- attempts to contact points by `(account_id, phone_id)`.
- attempts to accounts by `account_id`; to agents by `agent_id`.
- payments to attempts only through time ordering (no direct key).
- field_visits to addresses by `address_id`.

**Accept:** row counts match expectations (attempt_table equals 51,105 rows), no duplicate keys after joins, split column has no nulls.

---

### S6. EDA (produce figures and a written findings file)
Save plots to `outputs/eda/` and a `eda_findings.md` with a number for every claim.

Required analyses:
1. **Coverage:** attempts per contact point by source and priority_slot; share never tried; accounts with no attempts.
2. **Outcome mix:** network_response and disposition distributions overall and by channel (voice_bot vs tele_agent).
3. **Time:** answer rate and RPC rate by hour and day of week; by `salary_credit_day` proximity (days from salary day) for salaried accounts.
4. **Decay:** RPC rate against days since the contact point was added, and against number of prior failures.
5. **Gold vs behaviour:** for the 250 verified phones, box plots of answer rate, switched-off share, customer-hangup share, ring duration, `rpc` share, by `verified_status`. Identify which features separate third_party from borrower.
6. **Sources:** RPC and gold-borrower rates by `source` and `relation_recorded`.
7. **Experiment arm:** compare outcome rates in `random_contact_point` vs `rule_based`; show the selection-bias gap (rule-based always picks the same numbers).
8. **Shared numbers:** distribution of `n_accounts_sharing_phone_id`; outcomes for shared vs unshared phones.
9. **Text:** most frequent remark tokens by disposition; Hinglish/Kannada variants for "wrong number", "wife answered", "switched off".
10. **Skip traces:** hit rate by number of prior failed attempts, by source of the failed phones, by outstanding band; cost distribution.
11. **Payments vs contact:** share of payments that follow an RPC within 7 and 14 days versus none (separates contact-driven recovery from self-cure); payment channels.
12. **Field:** outcome mix by address_type, source, town; dwell time by outcome; photo reuse and its relation to outcome; check-in gap.
13. **Missingness:** table of the structural nulls with a check that their pattern matches the explanation in Section 1.
14. **Leakage hunt:** list any column that is only known after the outcome (e.g. `disposition` at time `t` must never be a feature for predicting `t`).

**Accept:** each finding in `eda_findings.md` cites a table or plot; decisions that change later stages (thresholds, features dropped) are listed explicitly.

---

### S7. Feature engineering (point-in-time)
**Function signature:** `build_cp_features(attempts, contact_points, accounts, as_of) -> DataFrame`, using only rows with `attempt_ts < as_of`.

Feature families (compute each over windows 7, 14, 30, 60 days and all-time):

**A. Outcome counts and rates**
- `n_attempts`, share of each `network_response`, share of each disposition group (rpc, third_party, wrong, no_contact).
- `answer_rate`, `rpc_rate`, `rpc_given_answer`, `third_party_given_answer`, `wrong_given_answer`.

**B. Recency**
- `days_since_last_attempt`, `days_since_last_answer`, `days_since_last_rpc`, `days_since_first_attempt`, `days_since_added`.
- `consecutive_failures` (current run of non-answers), `consecutive_off` (run of switched_off/not_reachable), `max_run_*`.

**C. Human-present evidence (key for avoiding vs invalid)**
- `customer_hangup_rate`, `fast_hangup_rate` (talk under 5s and answered), `busy_rejected_rate`, `ring_duration_mean/std/max`, `short_ring_rate`.

**D. Time-of-day / time-slot**
- Answer and RPC rate per slot (08-10, 10-12, 12-14, 14-18), with Beta-shrinkage toward the global slot rate to handle small samples. `best_slot`, `slot_entropy`.

**E. Decay-weighted rates (baseline family)**
- `beta_reach_mean`, `beta_reach_var` using weights `exp(-lambda * age_days)` as in S10.

**F. Context**
- `source`, `relation_recorded`, `priority_slot`, `is_reference`, `is_office`, `n_accounts_sharing_phone_id`.
- Agent and channel mix: share of bot calls, language-team match (agent `language_team` vs account `preferred_language`).

**G. Account-level cross-evidence (leave-one-out)**
- For each contact point: other contact points of the same account answered / got RPC in last 30 days (`other_cp_rpc_30d`), payments in last 30 days (`paid_recent`), payment via `upi_link` recently, `paid_other_lenders_30d`, DPD, bucket, `prev_ptp_broken`.
- Number of contact points on the account, and rank of this one.

**H. Graph features** from S8 (merge here).
**I. Text features** from S9 (merge here).

**Accept:**
- No NaN explosions (report null rate per feature; nulls are fine for tree models but must be intentional).
- **Leakage test passes** (S20).
- Feature list saved to `data/processed/feature_catalog.csv` with name, family, definition, and as-of rule.

---

### S8. Graph features
**Input:** `phone_account_links`, attempts. **Output:** `data/processed/graph_features.parquet`.

1. Bipartite graph: phone_id nodes and account_id nodes.
2. Features: `degree_phone = distinct accounts per phone_id`, `component_size` (union-find), `log1p(degree)`, `idf_weight = 1/log(1+degree)`.
3. Outcome sharing: for a shared phone_id, the **leave-one-out** RPC rate and wrong-number rate observed on the *other* accounts that use it (never include the target account's own outcome).
4. `is_reference_on_other_accounts`: how often this phone is recorded as a reference elsewhere.

**Accept:** component sizes sum to node count; leave-one-out excludes the target account (unit test).

---

### S9. Text features
**Input:** `remark` (normalised). **Output:** `data/processed/text_features.parquet`.

1. Build a keyword lexicon (lowercase, transliteration variants, Kannada script included) for these concepts: `wrong_number`, `new_owner` ("kisi aur ka", "naya number"), `family_answered` (wife, bhai, amma, appa, ...), `switched_off`, `busy`, `refusal`, `hardship`, `claims_paid`, `language_barrier`, `callback_request`, `abusive`.
2. Features: count of hits per concept per contact point over windows (point-in-time), plus a **character n-gram TF-IDF (2 to 5) + logistic regression** classifier for `third_party / wrong / recycled` trained on train-split only; output out-of-fold probabilities to avoid leakage.
3. Strip bot templates ("bot: RNR", "bot: wrong number") into a `remark_is_bot` flag so bot text does not dominate.

**Accept:** OOF text-model predictions exist for every train contact point; no text model trained on validation/test accounts.

---

### S10. Baselines (build first, must exist before any fancy model)
Report all baselines on the same evaluation harness (S18).

1. **Rule baseline:** the existing system: dial by `priority_slot` ascending; trace after 15 consecutive failures. Compute RPC per 1,000 dials and trace hit rate under this rule from the logs.
2. **Beta-Bernoulli with exponential decay** per contact point:

```
w_i   = exp(-lambda * (t_now - t_i))
alpha = alpha0 + sum_i w_i * y_i
beta  = beta0  + sum_i w_i * (1 - y_i)
p     = alpha / (alpha + beta)
var   = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
```

   `y_i` is `y_rpc` (primary) and `y_answered` (secondary). Tune `lambda`, `alpha0`, `beta0` on validation (log-loss).
3. **Global prior by source/relation** (for never-tried phones): empirical RPC rate per `(source, relation_recorded)`.

**Accept:** a baseline results table with AUC, log-loss, Brier, RPC per 1,000 dials, and verified-set accuracy.

---

### S11. Models

Two prediction tasks:

**Task 1 (per-attempt):** given a contact point and time slot, predict `P(rpc)` and `P(answered)` for the next attempt. Rows = attempts, features = S7 computed as-of the attempt time.
**Task 2 (per-phone state):** given the history to `data_end`, predict the state distribution. Rows = contact points, labels = weak soft labels, evaluation on gold.

#### S11a. Stage A: human present?
Binary LightGBM. Target: weak `human_present` (state in valid_* / third_party / recycled with an answered or rejected call) for tested phones. Objective `binary`, early stopping on validation log-loss.

#### S11b. Stage B: which state?
Multiclass LightGBM (`objective=multiclass`, `num_class=K`) trained with soft labels (use sample duplication weighted by soft-label probability, or a custom cross-entropy). Hierarchical combination: multiply by Stage A probability for human states.

#### S11c. Next-attempt model (Task 1)
Binary LightGBM per target (`rpc`, `answered`). This is the model used for dial ranking.

#### S11d. Hazard / survival (optional but recommended)
Person-period table: one row per contact point per day since first attempt; event = first day it enters {invalid, off_long_term, recycled_or_wrong} by weak labels; censored otherwise. Fit LightGBM or logistic regression on `event_next_day`. Output `S(t) = prod(1 - h(u))`. Be careful about censoring at `data_end`.

#### S11e. HMM state tracking
- States: `valid_reachable, valid_avoiding, temp_unreachable, off_long_term, third_party, recycled_or_wrong, invalid`.
- Observation symbol per attempt = (`network_response`, disposition group) combined into a small discrete alphabet.
- Initialise emission matrix **supervised** from weak-labelled contact points (counting), transitions from observed state changes (or hand-set sticky matrix with diagonal about 0.9), then optionally refine with EM.
- Forward filtering for the live posterior:

```
alpha_t(j) ∝ b_j(o_t) * sum_i alpha_{t-1}(i) * A_ij
```

- For irregular gaps, use `A(dt) = expm(Q * dt)` with a rate matrix `Q`.
- Output: filtered state posterior per contact point as of `data_end` (and as of any `as_of`).

**Accept:**
- Probabilities of each model sum to 1 (multiclass) and lie in [0, 1].
- Model comparison table on validation: baseline vs Stage A/B vs +hazard vs +HMM.
- Feature importance (gain) and SHAP summary saved.
- Models saved to `outputs/models/` with the feature list and config.

---

### S12. Recycled / third-party detector
**Goal:** very high recall on {recycled_or_wrong, third_party}, since calling them risks disclosure.

1. **CUSUM** per contact point on a daily log-likelihood ratio between "same subscriber" and "new subscriber" outcome distributions (estimate both from weak-labelled data).

```
S_t = max(0, S_{t-1} + log(p1(x_t) / p0(x_t)));  alarm if S_t > h
```

2. Text-model probability (S9) and graph signals (S8).
3. Combine with noisy-OR: `P = 1 - prod(1 - p_i)`, or stack with logistic regression on OOF predictions.
4. **Threshold selection:** choose `tau` on the validation set so that recall on {recycled_or_wrong, third_party} against **gold** labels is at least `recycled_target_recall` (0.99 if feasible, otherwise report the best attainable and the false-positive rate it costs). Report precision at that threshold.
5. Output `suppress_flag` and `suppress_prob` per contact point.

**Accept:** recall/FPR table on gold (train/val), the chosen threshold stored in config, and an explicit statement of how many valid borrower numbers get suppressed by mistake.

---

### S13. Calibration
1. Fit on validation predictions (or OOF predictions on train), **never on test**.
2. Per target, compare Platt (`sigmoid`) vs isotonic. Choose by validation ECE and Brier. Use Platt if fewer than about 1,000 calibration points.
3. For multiclass, calibrate one-vs-rest then renormalise, or use temperature scaling.
4. Report reliability diagrams (10 bins) and ECE before and after.

```
ECE = sum_m (|B_m|/n) * |acc(B_m) - conf(B_m)|
Brier = mean((p - y)^2)
```

**Accept:** ECE after calibration is less than before, saved calibrators, reliability plot.

---

### S14. Tuning
**Framework:** Optuna, 50 to 100 trials per model, seed fixed, sampler TPE.
**Validation scheme:** `GroupKFold(n_splits=5)` grouped by `account_id` **within the train split only**; the given validation split is the early-stopping and final model-selection set. Test split is not touched.

Search spaces (LightGBM):
- `learning_rate` 0.01 to 0.1 (log), `num_leaves` 8 to 64, `max_depth` 3 to 8, `min_child_samples` 20 to 200, `feature_fraction` 0.5 to 1.0, `bagging_fraction` 0.5 to 1.0, `lambda_l1` and `lambda_l2` 1e-3 to 10 (log), `n_estimators` up to 2000 with early stopping 100.
- Class weights or `scale_pos_weight` for imbalanced targets.
- Optional **monotone constraints**: more days since last answer should not raise reachability; more consecutive failures should not raise it.

Objectives: minimise validation log-loss for probabilities, and report PR-AUC for rare classes (third_party, recycled, invalid).

Also tune (cheap grid or Optuna):
- Decay `lambda` and Beta priors (S10).
- HMM transition stickiness and CUSUM `h`.
- Weak-label rule thresholds (S4), using train-split gold only.
- Decision thresholds (S15).

**Accept:** a tuning log with best params, trial history, and the check that validation score beats default parameters. No tuning against test.

---

### S15. Decision policy and skip-trace valuation

#### S15a. Action policy (rule table over calibrated posteriors)
Apply **compliance overrides first** (suppress if `suppress_flag`), then:

| Condition | Action |
|---|---|
| suppress_flag | `suppress` (never discuss the debt) |
| best phone `P(reachable)` above threshold | `dial` best phone at its best slot |
| best phone `valid_avoiding` high | `switch_channel` (WhatsApp / SMS / field) |
| best phone `temp_unreachable` high | `retry_with_backoff` |
| `off_long_term` or `invalid` high on all phones | `try_next_contact_point`, else `trace` |
| address states (S17) | `visit`, `change_visit_time`, `resolve_location`, or `trace` |

Attach `reason_codes`: TreeSHAP top 3 features mapped to templated phrases.

#### S15b. Trace value (EVT)
For each account that is trace-eligible:

```
EVT = P(find) * P(recover | reached) * Outstanding_or_Overdue * uplift - C_trace
```

- `P(find)`: LightGBM on skip_traces (766 rows) with features **as of `trace_date`** (point-in-time), target `found_new`. Use cross-validation and report AUC; with only 766 rows keep the model small.
- `P(recover | reached)`: share of accounts with a payment within 30 days after a first RPC, modelled with a small GBDT or logistic regression on account features.
- `uplift`: difference in recovery between accounts that were reached and comparable accounts that weren't. Estimate with a T-learner, and report that confounding exists (state it).
- `C_trace`: `cost_inr` (mean by result type, default 102).
- Rank by `EVT / C_trace` and pick greedily within a budget. Compare against the 15-consecutive-failures rule on validation: hit rate and expected recovered amount per rupee.

**Accept:** EVT ranking beats the baseline rule on validation (or the report states honestly that it doesn't, and why), and the policy table is covered by tests.

---

### S16. Exploration and off-policy evaluation (OPE)
1. Use `selection_propensity` as the logging propensity `mu(a|x)`.
2. Build an evaluation set from accounts in the `random_contact_point` arm (and propensity-weighted rule-based rows).
3. Evaluate a candidate dial policy `pi` (e.g. "pick the contact point with the highest predicted `P(rpc)`"):

```
V_IPS = (1/n) * sum_i r_i * pi(a_i|x_i) / mu(a_i|x_i)
V_DR  = (1/n) * sum_i [ q_hat(x_i, pi(x_i)) + (pi(a_i|x_i)/mu(a_i|x_i)) * (r_i - q_hat(x_i, a_i)) ]
```

   `r_i = y_rpc`. Clip weights at 10 to 20. Report mean and bootstrap 95% CI. With about 123 random-arm accounts, expect wide intervals; say so.
4. Exploration design for production: Thompson sampling, `theta ~ Beta(alpha, beta)` per candidate, budget `explore_budget_share`; never explore suppressed contact points. Log the propensity.

**Accept:** IPS and DR estimates for baseline vs model policy with CI, clipping stated.

---

### S17. Address module (phase 2)
**Labels** from `field_visits.outcome` aggregated per address (point-in-time):

| outcome | address state |
|---|---|
| met_borrower, cash_collected | valid_occupied |
| met_family | valid_family_present |
| locked_premises | valid_absent_usually |
| neighbour_says_shifted | moved |
| no_such_person | wrong_or_fabricated |
| address_not_traceable | hard_to_find |

**Features:** address_type, source, `town_id == OUT`, `text_len`, `has_kannada_script`, landmark flags, pincode, lender `kyc_address_format`, visit count, previous outcomes, `dwell_s` mean, `gps_accuracy_m`, `travel_gap_s`, `photo_reused`, agent id and tenure, days since added.
**Hierarchical shrinkage** toward pincode or town group g: `p_hat = (successes + kappa * mu_g) / (visits + kappa)`, with `kappa` fitted by empirical Bayes (Beta-Binomial maximum likelihood).
**Model:** multiclass LightGBM + calibration, same pattern as S11 and S13. Treat addresses never visited as unlabeled.

**Accept:** per-state precision/recall on validation, calibration plot, reason codes.

---

### S18. Evaluation and reporting (touch test once)
Run all baselines and final models on **test** once.

Metrics (per contact-point type, overall and by `portfolio`/`lender`):
- AUC, PR-AUC, log-loss, Brier, ECE.
- Gold-set metrics on the 36 test verified phones (with a bootstrap CI, because it is tiny).
- RPC per 1,000 dial attempts: baseline order vs model order (simulated on test accounts).
- Wasted attempts on invalid / off-long-term contact points before action.
- Trace hit rate and expected recovered amount per INR spent: EVT vs 15-failures rule.
- Recycled/third-party recall at the chosen threshold, and suppression of valid borrowers.
- OPE estimates from S16.
- **Ablation table:** baseline, + GBDT, + hazard, + HMM, + graph, + text, + EVT. Report the delta at each step.

Deliver `outputs/report.md` plus all figures and a one-page summary.

---

### S19. Daily pipeline and output contract
`s19_daily_run.py` re-runs ingestion, features as of today, scoring, calibration, policy and trace ranking, and writes:

- `outputs/contact_point_scores.parquet`: `cp_id`, `account_id`, `phone_id`/`address_id`, `p_rpc`, `p_answered`, state probabilities (one column per state), `suppress_flag`, `suppress_prob`, `best_slot`, `score_date`, `reason_codes`.
- `outputs/account_actions.parquet`: `account_id`, `action`, `target_cp_id`, `channel`, `reason_codes`, `explore_flag`, `propensity`.
- `outputs/trace_queue.parquet`: `account_id`, `evt`, `evt_per_inr`, `p_find`, `rank`.
- `outputs/suppression_list.parquet`: `cp_id`, `suppress_prob`, `reason`.

**Accept:** one command runs end to end; schemas validated with assertions; runtime logged; results deterministic given the seed.

---

## 4. S20. Tests and checks (write these early)

1. **Leakage test:** compute features for a contact point at `as_of=t`; delete all attempts with `ts >= t` from the raw data and recompute. The features must be identical. Run it for a random sample of 200 contact points and several `t` values.
2. **Split integrity:** no `account_id` appears in more than one split; test set unused before S18.
3. **Gold isolation:** no verified phone's label appears in training rows or in rule thresholds for validation/test accounts.
4. **Join correctness:** row counts unchanged after every join; `cp_id` unique.
5. **Probability sanity:** all probabilities in [0, 1]; multiclass rows sum to 1.
6. **Determinism:** two runs with seed 42 give identical outputs.
7. **Leave-one-out graph test:** target account's own outcomes are excluded.
8. **Policy test:** a `suppress_flag` contact point never appears as a `dial` target.
9. **Monotone sanity:** longer time since last answer does not increase `p_rpc` (check on synthetic perturbation).

---

## 5. Pitfalls checklist

- Treating `answered` as RPC. Third-party numbers have the highest answer rate.
- Using `phone_masked` as a key. It is only the last 4 digits.
- Treating never-tried contact points as negative.
- Training only on `rule_based` calls and inheriting the old dialler's bias; use the random arm and propensities for evaluation.
- Using the disposition of the current attempt as a feature for predicting that attempt.
- Using skip-trace features computed after `trace_date`.
- Random row-level splits instead of `splits.csv`.
- Tuning or choosing thresholds on the test set.
- Mean-imputing structural nulls instead of using flags.
- Using payments after an attempt to explain that attempt (payments are an outcome, usable only as past evidence at a later `as_of`).
- Reporting gold-set metrics without confidence intervals; with 36 test phones the interval is wide.

---

## 6. Suggested order of implementation

`S0 → S1 → S2 → S3 → S4 → S5 → S6 → S10 → S7/S8/S9 → S11 → S12 → S13 → S14 → S15 → S16 → S17 → S18 → S19`, with S20 tests written alongside S7 and S4.

Minimum viable submission: S0 to S7, S10, S11 (Stage A/B + next-attempt model), S13, S15 (policy + EVT), S18. Add S12, S16, S17 as time allows.
