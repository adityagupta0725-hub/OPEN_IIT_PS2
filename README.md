# PS2: Right-Party Contact (RPC) Prediction and Skip-Trace Prioritisation

> Source: CreditNirvana Problem Statement 2 (AI-native debt collections, India). This README is the implementation blueprint: architecture, repo layout, models, and the maths each module must implement.

> **⚠️ IMPORTANT RUN INSTRUCTIONS:** The `modelling.ipynb` notebook contains the core mathematical pipeline, but **it should NOT be run manually or directly**. It is designed to be dynamically patched and executed by the automated utility scripts (`patch_notebook.py`, `patch_notebook2.py`, and `run_modelling.py`) which inject new experimental code (like `hmmlearn` Expectation-Maximization and hierarchical intents) automatically before execution.

---

## 1\. Problem summary

For every **phone number** and **address** on file, predict the probability that it reaches the borrower (a **right-party contact**, RPC). A decision layer then chooses, per account, one of:

`continue` | `switch contact point` | `switch channel` | `skip-trace` | `suppress`

Skip-traces are ranked by the **expected recovery value** of finding the borrower, not by a fixed attempt count.

### Hard requirements (from the problem statement)

| Requirement | Implication for design |
| :---- | :---- |
| Unit of prediction is the **contact point**, not the account | Event table keyed by `(contact_point_id, attempt)` |
| "Avoiding" vs "invalid" must be separated | Evidence-of-live-human features \+ hierarchical classifier |
| True labels are never observed | Weak supervision \+ PU learning, soft labels |
| Health changes over time | Decay / hazard / HMM state tracking |
| Recycled numbers risk debt disclosure | Asymmetric threshold (high recall), suppression list, **0 disclosure target** |
| Trace is a value-of-information decision | EVT formula \+ budgeted greedy selection |
| Feedback loop from own dial order | Exploration budget \+ logged propensities |
| Scores refreshed **daily** | Batch pipeline, incremental features |
| Explainable and auditable | Reason codes via SHAP, hard rule layer |
| Out of scope | In-call identity verification |

### Phasing

- **Weeks 1-6:** phone numbers (health score, avoiding-vs-invalid, recycled check, dialer and suppression integration).  
- **Weeks 7-12:** addresses and field evidence, PS3 location confidence, account-level trace decision across all contact points.

---

## 2\. Acronyms

| Acronym | Meaning |
| :---- | :---- |
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

## 3\. System architecture (Implemented Pipeline)

The system is a layered probabilistic pipeline. Each layer feeds the next:

\[1\] Ingestion \+ event table
        |
        v
\[2\] **Layer 1: Source–Relation Prior** (α₀, β₀ from group train rates)
        |
        v
\[3\] **Layer 2: Beta–Bernoulli with Exponential Decay** (causal rolling rates with temporal decay λ=0.05/day)
        |
        v
\[4\] **Layer 3: Logistic Regression (RPC)** (P(RPC | history, time, account) — acts as the grand integrator)
        |
        v
\[5\] **Layer 4: Logistic Regression (Repayment)** (P(pay within 14d | RPC) using account features)
        |
        v
\[6\] **Layer 5: Continuous-Time HMM** (Tracks 5 hidden phone health states using matrix exponentials)
        |
        v
\[7\] **Account Roll-Up & Trace Valuation** (EVT gating)
        |
        v
\[8\] **Outputs:** dial list order, action \+ reason code, trace queue, suppression list

---

## 4\. Proposed repository structure

ps2-rpc/

|-- README.md

|-- requirements.txt

|-- configs/

|   |-- default.yaml            \# decay lambda, thresholds, budgets, costs

|   \`-- states.yaml             \# state taxonomy \+ allowed transitions

|-- data/

|   |-- raw/                    \# real data if provided

|   \`-- synthetic/              \# simulator output

|-- src/

|   |-- simulate/

|   |   \`-- simulator.py        \# hidden-state generator for contact points

|   |-- ingest/

|   |   \`-- events.py           \# build (contact\_point, attempt) event table

|   |-- labels/

|   |   |-- labeling\_functions.py

|   |   |-- label\_model.py      \# weak supervision combiner

|   |   \`-- pu.py               \# Elkan-Noto correction

|   |-- features/

|   |   |-- telephony.py

|   |   |-- text.py             \# Hinglish remark/transcript features

|   |   |-- graph.py            \# shared-contact graph

|   |   \`-- build.py

|   |-- models/

|   |   |-- baseline\_beta.py    \# Beta-Bernoulli with decay

|   |   |-- gbdt\_human.py       \# stage A

|   |   |-- gbdt\_state.py       \# stage B

|   |   |-- hazard.py           \# discrete-time survival

|   |   |-- hmm.py              \# forward algorithm, covariate transitions

|   |   |-- recycled.py         \# CUSUM \+ text \+ noisy-OR

|   |   \`-- calibrate.py        \# Platt / isotonic

|   |-- decision/

|   |   |-- policy.py           \# action rule table \+ compliance overrides

|   |   |-- trace\_value.py      \# EVT \+ greedy budgeted selection

|   |   \`-- explore.py          \# Thompson sampling \+ propensity logging

|   |-- explain/

|   |   \`-- reason\_codes.py     \# SHAP \-\> plain-language reasons

|   |-- eval/

|   |   |-- metrics.py          \# AUC, Brier, ECE, RPC per 1k dials

|   |   |-- ope.py              \# IPS / DR off-policy evaluation

|   |   \`-- backtest.py         \# wasted-attempts simulation vs fixed rule

|   \`-- pipeline/

|       \`-- daily\_run.py        \# end-to-end daily batch

|-- notebooks/

|   \`-- 01\_eda.ipynb ... 05\_results.ipynb

\`-- tests/

---

## 5\. Data model

### 5.1 Event table (one row per attempt)

| Column | Description |
| :---- | :---- |
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

Phone: `valid_reachable`, `valid_avoiding`, `temp_unreachable`, `off_long_term`, `recycled`, `third_party`, `invalid`. Address: `valid_occupied`, `valid_absent_usually`, `moved`, `hard_to_find`, `fabricated_incomplete`.

---

## 6\. Modules, models and maths

### 6.1 Synthetic simulator (`simulate/`)

Needed because real data may be limited and for ground-truth testing.

- Sample a hidden state sequence per contact point from a transition matrix `A`.  
- Emit attempt logs from state-conditional emission distributions (answer rate by hour slot, ring duration, network response, hangup speed).  
- Inject: shared numbers across borrowers, recycled events (long-off gap followed by wrong-number dispositions), untested contact points, and noisy agent logging (a fraction of RPCs unlogged).  
- Output both observations and the true state, so every later layer can be scored against truth.

### 6.2 Label engine (`labels/`)

**Labelling functions** (rules) map evidence to a state or abstain:

| Evidence | Label |
| :---- | :---- |
| RPC logged and payment or PTP followed | `valid_reachable` |
| Network `does_not_exist` | `invalid` |
| Long run of `switched_off` | `off_long_term` |
| "wrong number" / "galat number" disposition or remark | `recycled` or `third_party` |
| Answered by relative | `third_party` |
| Never tested | **unlabeled** (never treated as negative) |

**Label model (naive Bayes combiner):**

log\[ P(y=1 | lambda) / P(y=0 | lambda) \] \= prior\_logodds \+ sum\_k w\_k \* lambda\_k

w\_k \= log( acc\_k / (1 \- acc\_k) )

Rule accuracies `acc_k` are estimated without ground truth (Snorkel-style).

**PU correction (Elkan-Noto)** for partially labelled positives (`s=1` means labelled):

P(y=1 | x) \= P(s=1 | x) / c,     c \= P(s=1 | y=1)   (estimated on held-out known positives)

**Soft-label training loss:**

L \= \- sum\_k q\_k \* log p\_k           (q \= soft label vector, p \= model softmax)

### 6.3 Feature engineering (`features/`)

Telephony features:

- `answer_rate_slot[h] = answers_h / attempts_h` for each hour slot h  
- `hangup_fast_rate` \= share of answered calls shorter than \~3s  
- `ring_duration_mean`, `ring_duration_std`  
- `days_since_last_RPC`, `days_since_last_answer`  
- `consecutive_unreachable_count`  
- network response counts by type (last 7/30/90 days)

**Cross-channel evidence (key for avoiding vs invalid):**

- `engaged_other_channel_30d` (payment-link open, WhatsApp read, answered another number)

Text features (Hinglish): character n-gram TF-IDF (n \= 2..5) over remarks and transcripts. Character n-grams absorb spelling variation ("band hai", "bnd hai").

Record features: `source`, `age_days`, `ever_confirmed_by_payment`.

### 6.4 Shared-contact graph (`features/graph.py`)

Bipartite graph: borrowers on one side, contact points on the other.

- Degree: `d(c)` \= distinct borrowers linked to contact point c. Use `log(1 + d(c))`.  
- Connected components via union-find (near-linear time).  
- IDF-style down-weighting: `w(c) = 1 / log(1 + d(c))`.  
- Fraction of links where the number is a reference or co-borrower, not a primary contact.

A number tied to many unrelated borrowers suggests a reference, shared, or recycled contact point.

### 6.5 State estimator (Implemented Models)

The actual implemented state estimator follows a layered probabilistic approach rather than raw GBDTs.

#### 6.5.1 Layer 1 & 2: Bayesian Prior and Beta-Bernoulli with Exponential Decay

weight\_i \= exp( \-lambda \* (t\_now \- t\_i) )
alpha \= alpha\_0 \+ sum\_i weight\_i \* s\_i
beta  \= beta\_0  \+ sum\_i weight\_i \* (1 \- s\_i)
P(reachable) \= alpha / (alpha \+ beta)

Confidence falls automatically as evidence ages. `lambda` is set to 0.05 per day. This provides pure call-history probability (`p_beta`).

#### 6.5.2 Layer 3: Logistic Regression (Next-Dial RPC Integrator)

P(RPC | x) \= sigmoid( w . x\_t \+ b )

This layer integrates the `p_beta` history with time-of-day features, categoricals, and account debt profile to produce the final probability of reaching the borrower on the next dial. Fitted via L-BFGS with L2 regularization.

#### 6.5.3 Layer 4: Logistic Regression (Repayment Probability)

P(pay within 14d | RPC) \= sigmoid( w\_r . x\_account \+ b\_r )

Predicts the probability of actual payment conditional on reaching the borrower. Uses only account-level financial features (outstanding, emi_amount, etc.).

#### 6.5.4 Layer 5: Continuous-Time Input-Output HMM

- **Hidden states**: 5 phone health states (valid_reachable, valid_avoiding, temp_unreachable, off_long_term, invalid).
- **Continuous time transitions**: Irregular gaps between attempts use continuous time `A(dt) = expm( Q * dt )`, where Q is a rate matrix (`scipy.linalg.expm`).  
- **Forward filtering** (live belief over states):
  alpha\_t(j)  proportional to  b\_j(o\_t) \* sum\_i alpha\_{t-1}(i) \* A\_ij
- **Output**: filtered posterior `alpha_t(.)` representing the probability of the phone being in each of the 5 states today. This feeds directly into the decision policy layer.

#### 6.5.4 Recycled-number detector (`models/recycled.py`)

Signature: history of RPCs, then a quiet or off period, then a shift to wrong-number dispositions or "I don't know this person". Telecom operators reallocate inactive numbers after a quiet period (commonly cited as about 90 days in India; verify before hardcoding).

**CUSUM change detection:**

S\_t \= max( 0, S\_{t-1} \+ log\[ p1(x\_t) / p0(x\_t) \] )

alarm when S\_t \> h

`p0` \= outcome likelihood under "same subscriber", `p1` \= under "new subscriber".

**Text classifier:** character n-gram TF-IDF with logistic regression, detecting Hinglish phrases such as "galat number", "yahan koi nahi rehta", "yeh kisi aur ka number hai".

**Graph signal:** number shared by many unrelated borrowers.

**Combine detectors (noisy-OR):**

P(recycled) \= 1 \- prod\_i ( 1 \- p\_i )

**Asymmetric decision:** choose the action minimising expected loss.

a\* \= argmin\_a  sum\_s P(s) \* L(a, s)

L(dial, recycled) \>\> L(suppress, valid)

In practice, pick threshold `tau` on the validation ROC so that **recall \>= 99%** and report the false-positive rate it costs. Suppressed numbers can be re-verified through a safe channel.

#### 6.5.5 Calibration (`models/calibrate.py`)

Platt:     p\_cal \= sigmoid( a \* score \+ b )

Isotonic:  fit monotone non-decreasing f minimising sum\_i ( y\_i \- f(s\_i) )^2   (PAV algorithm)

Calibrate per contact-point type and per portfolio. Validate on a **temporal split grouped by borrower** (never a random split).

Brier \= (1/n) \* sum\_i ( p\_i \- y\_i )^2

ECE   \= sum\_m ( |B\_m| / n ) \* | acc(B\_m) \- conf(B\_m) |      (probability bins B\_m)

AUC   \= P( score of random positive \> score of random negative )

### 6.6 Decision policy (`decision/policy.py`)

Rule table over calibrated posteriors, with a **hard compliance override layer applied first**.

| Condition | Action |
| :---- | :---- |
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

phi\_i \= sum\_{S subset of F \\ {i}}  \[ |S|\! \* (M \- |S| \- 1)\! / M\! \] \* \[ f(S U {i}) \- f(S) \]

Map the top 2-3 features to templated phrases, for example: "Answers 9-11am, hangs up within 2s, so likely avoiding. Recommend WhatsApp."

### 6.8 Skip-trace valuation (`decision/trace_value.py`)

EVT \= P(trace finds valid CP) \* P(recover | reached) \* Outstanding \* uplift \- C\_trace

- `uplift` \= recovery gain over not tracing.  
- `P(trace finds valid CP)`: separate GBDT trained on historical trace outcomes.  
- Trace only if `EVT > 0`.  
- Under a budget B, sort by `EVT / C_trace` and select greedily until B is spent (greedy fractional-knapsack approximation).  
- Only trace accounts that are findable but not currently reachable.

### 6.9 Exploration (`decision/explore.py`)

Reserve about 2-5% of attempts for exploration.

- **Thompson sampling:** for each candidate contact point with posterior `Beta(alpha, beta)`, draw `theta ~ Beta(alpha, beta)` and rank by `theta`.  
- Never explore suppressed recycled or third-party numbers. Compliance beats exploration.  
- **Log propensity** `mu(a | x)` for every action taken.

### 6.10 Address module (phase 2\)

Same architecture, different features:

- "locked premises" vs "nobody of that name here" outcomes  
- GPS dwell at check-in, visit-time patterns  
- PS3 location confidence as an input feature

Field data is sparse, so use **hierarchical shrinkage** toward the pincode or locality group g:

p\_hat\_addr \= ( successes \+ kappa \* mu\_g ) / ( visits \+ kappa )

`mu_g` is the group mean success rate. `kappa` is the prior strength (pseudo-visits), fitted by **empirical Bayes**, i.e. maximum likelihood of the Beta-Binomial marginal across addresses.

---

## 7\. Evaluation (`eval/`)

### 7.1 Protocol

- **Temporal split, grouped by borrower.** Train on earlier periods, validate and test on later ones. All contact points of a borrower stay in the same fold.  
- Report per contact-point type (phone / address) and per portfolio.

### 7.2 Metrics

| Metric | Definition |
| :---- | :---- |
| AUC, Brier, ECE | As in 6.5.5 |
| RPC per 1,000 dial attempts, per 100 visits | Compared with the current rules |
| Wasted attempts on invalid contact points before action | Backtest vs fixed attempt-count rule |
| Skip-trace hit rate, recovery from traced accounts | Compared with fixed-count rule |
| Recycled-number recall at fixed FPR | Safety metric |
| Third-party disclosure incidents | Target: zero |

### 7.3 Off-policy evaluation (using exploration logs)

V\_IPS \= (1/n) \* sum\_i r\_i \* pi(a\_i | x\_i) / mu(a\_i | x\_i)

V\_DR  \= (1/n) \* sum\_i \[ q\_hat(x\_i, pi(x\_i))

                       \+ ( pi(a\_i | x\_i) / mu(a\_i | x\_i) ) \* ( r\_i \- q\_hat(x\_i, a\_i) ) \]

`q_hat` is a learned reward model. Clip importance weights (for example at 10-20) to control variance.

### 7.4 Ablation story to present

Beta-decay baseline, then plus GBDT, then plus hazard, then plus HMM, then plus graph and cross-channel features, then plus EVT trace ranking. Show measured lift at each step.

---

## 8\. Compliance and safety design

- **Hard rule layer** sits above all models and cannot be overridden by scores.  
- Recycled and third-party numbers go to a **suppression list**. The debt is never discussed with the wrong person (RBI Fair Practices Code).  
- Bureau and reference data is used only for its collected purpose (DPDP Act).  
- Skip-trace sources must be lawful and auditable.  
- Every action has a reason code and a logged score, so it is auditable.  
- Scores refresh **daily** before outbound planning.

---

## 9\. Build order

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

## 10\. Libraries

| Need | Library |
| :---- | :---- |
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

---

## 11\. Open questions to confirm with organisers

- Exact **RPC definition** (does an answer by the borrower with no payment intent count?).  
- How **never-tested** contact points are represented in the data.  
- Cost values: `C_trace`, cost per dial, cost per field visit.  
- Whether any **telecom signals** (HLR lookup, operator reallocation data) are available.  
- Whether real data or sandbox access will be provided, and the format.