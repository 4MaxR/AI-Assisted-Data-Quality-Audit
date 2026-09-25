# Data Quality Audit — Telecom Customer Churn Dataset

**Prepared for:** Data quality review before any downstream churn analytics
**Dataset:** `telecom-customer-churn-data-quality-challenge.zip` (3 CSVs, 5.1 MB)
**Rows analysed:** 10,200 + 120,000 + 10,000
**Audit date:** 2026-09-25

> **Companion document.** The narrative case study — findings, charts, and the method
> explained — is the [project README](../README.md). This document is the formal audit
> record: all 11 issues with per-issue evidence, and the repair rule for each. The
> README is self-contained; this is the depth behind it.

---

## Navigation

| § | Section | Answers |
|---|---|---|
| [1](#1-executive-summary) | Executive summary | What did the audit conclude? |
| [2](#2-scope-and-method) | Scope and method | What was tested, and how? |
| [3](#3-data-inventory-and-grain) | Data inventory and grain | What is in the dataset, and does it join? |
| [4](#4-what-is-clean-stated-explicitly) | What is clean | Which checks passed? |
| [5](#5-findings-register) | Findings register | Every issue, with evidence and repair rule |
| [6](#6-business-impact-of-not-cleaning) | Business impact of not cleaning | What the raw data gets wrong |
| [7](#7-limitations-and-uncertainty) | Limitations and uncertainty | What this audit cannot claim |
| [8](#8-recommendations-in-priority-order) | Recommendations | What to do, in order |
| [9](#9-deliverables) | Deliverables | What was produced, and where |

**Jump straight to a severity band:** [🔴 Critical](#severity-critical-blocks-the-stated-use-case) · [🟠 High](#severity-high) · [🟡 Medium](#severity-medium) · [🔵 Low](#severity-low)

**Jump to a specific issue:** [DQ-11](#dq-11--churn_labelschurn--the-label-is-uninformative) · [DQ-01](#dq-01--customer_info--195-exact-duplicate-records) · [DQ-02](#dq-02--customer_infosignupdate--5-conflicting-duplicates) · [DQ-03](#dq-03--customer_infoage--3500-missing) · [DQ-04](#dq-04--customer_infomonthlycharges--49-values-from-a-contaminant-distribution) · [DQ-05](#dq-05--customer_infomonthlycharges--5-negative-charges) · [DQ-07](#dq-07--usage_data--customer_info--2278-rows-dated-before-signup) · [DQ-06](#dq-06--customer_infomonthlycharges--24-implausibly-low-charges) · [DQ-09](#dq-09--usage_datasmscount--no-behavioural-structure) · [DQ-10](#dq-10--usage_data--callminutes-and-datausagegb-are-gaussian-synthetic-draws) · [DQ-08](#dq-08--usage_datadatausagegb--58-rows-censored-at-exactly-0)

---

## 1. Executive summary

The dataset is **structurally sound but semantically unreliable**. Referential
integrity across the three tables is perfect, the usage table is a complete
`customer × month` grid with no duplicates, and all categorical fields are
clean. The problems are concentrated in *values*, not in *shape*.

Eleven issues were found. Two change how the data must be used:

| | Finding | Why it matters |
|---|---|---|
| **1** | **The churn label has no measurable relationship with any feature.** Three model families all score 0.516–0.518 cross-validated ROC-AUC against a random baseline of 0.500 — inside a shuffled-label noise band that reaches 0.537. | A predictive churn model cannot be built from this data. Any model will look like it "works" on a train/test split by luck; it will not generalise. |
| **2** | **2,278 usage rows (1.90%, across 751 customers) are dated before the customer signed up.** | Any tenure, lifetime-value, or time-to-churn calculation is wrong for 7.5% of the customer base until the source of truth is fixed. |

Everything else is repairable with documented, auditable rules — 200 duplicate
customer rows, 3,500 missing ages, 78 invalid charge values, 58 censored data
usage rows.

**Recommended action:** use the dataset for *coverage and quality work*, not as
a churn-modelling benchmark. Fix the timeline defect at source before publishing
anything derived from tenure.

---

## 2. Scope and method

I audited all three tables as delivered, then tested whether the data supports
the purpose implied by its name. Five sequential passes, all reproducible
(`scripts/01`–`08`):

1. **Structural profiling** — grain, dtypes, missingness, raw value ranges
2. **Targeted checks** — duplicates, out-of-range values, timeline logic, hygiene
3. **Anomaly characterisation** — testing *what kind* of defect each anomaly is
4. **Distribution forensics** — is this real data or synthetic noise?
5. **Rule definition and repair** — one documented rule per defect, with impact measurement

Design principle: **nothing was silently fixed.** Invalid values became `NULL`
behind an explicit flag column, so every touched row stays visible.

Reference thresholds used: ±3 standard errors for missingness randomness, IQR
fences for first-pass outlier detection, and BIC for distribution mixture
selection.

---

## 3. Data inventory and grain

| Table | Grain | Rows | Verdict |
|---|---|---|---|
| `customer_info.csv` | 1 row per customer | 10,200 | **10,000 unique IDs** — 200 surplus rows |
| `usage_data.csv` | 1 row per customer per month | 120,000 | Clean: 10,000 × 12, complete grid |
| `churn_labels.csv` | 1 row per customer | 10,000 | Clean: unique, binary, 7.86% positive |

**Referential integrity is perfect.** Zero orphan records in any direction:

- customers in `customer_info` with no usage: **0**
- usage rows with no customer record: **0**
- customers with no churn label: **0**
- labels with no customer record: **0**

**Usage table hygiene is excellent.** All 12 months present and contiguous
(2023-01 → 2023-12), every value is a true end-of-month date, zero duplicate
`(CustomerID, Month)` pairs, and every customer has exactly 12 rows. This is the
highest-quality part of the dataset.

**Churn label distribution is plausible:** 786 churners / 10,000 = 7.86%. No
missing labels, no out-of-range values.

---

## 4. What is clean (stated explicitly)

An audit that only lists problems is misleading. The following were tested and
passed:

- Referential integrity across all three tables (see above)
- Usage table grain, completeness, and duplicate-freedom
- `Month` is uniformly formatted, end-of-month, no gaps
- `Gender`, `Region`, `ContractType` — no whitespace, no casing variants, no
  unexpected categories (2 / 3 / 3 distinct values, all valid)
- `Age` — where present, always an integer in 18–69. No zeros, no 999/–1
  sentinels, no impossible values. Missingness is the *only* problem.
- `SignupDate` — 100% parseable, no malformed dates, no future dates
- `Churn` — no missing, no duplicates, plausible base rate

---

## 5. Findings register

Full machine-readable version: `outputs/data_quality_register.csv`

### Severity: Critical (blocks the stated use case)

#### DQ-11 · `churn_labels.Churn` — the label is uninformative
**Affected:** all 10,000 rows

5-fold cross-validated ROC-AUC:

| Model | CV ROC-AUC | vs random |
|---|---|---|
| Logistic regression | 0.5176 | +0.0176 |
| Random forest (500 trees) | 0.5170 | +0.0170 |
| Histogram gradient boosting | 0.5164 | +0.0164 |
| **Shuffled-label null (20 draws)** | **0.4708 – 0.5372** | — |

The real models score **inside the null band produced by randomly permuting the
labels**. Single-feature tests agree — every candidate predictor returns
p > 0.11 except `SMSCount` (p = 0.026), which is 1 of 10 tests and therefore
expected by chance.

Consistent with this, `ContractType` shows Yearly contracts churning *most*
(8.82%) versus Monthly (7.56%) — backwards from real telecom behaviour, where
long commitments reduce churn.

**Interpretation.** The label appears to have been assigned independently of the
features. This is consistent with a synthetic dataset where the quality defects
are the exercise, not the prediction.

**Rule:** do not present, sell, or benchmark a predictive model on this data.
Report the label as an uninformative target.

---

### Severity: High

#### DQ-01 · `customer_info` — 195 exact duplicate records
**Evidence:** 10,200 rows carry only 10,000 unique `CustomerID`s. 195 IDs
appear twice with byte-identical rows across every column.

**Impact:** a naive join to `churn_labels` returns 10,200 rows and silently
double-counts 195 customers — see §6.

**Rule:** drop duplicates, keep first occurrence. ✅ auto-fixed

#### DQ-02 · `customer_info.SignupDate` — 5 conflicting duplicates
**Affected:** CUST_248, CUST_308, CUST_333, CUST_391, CUST_466 (10 rows)

Each of these 5 IDs appears twice with **every field identical except
`SignupDate`** — and in all 5 cases the two dates are an exact **month/day swap**:

| ID | Date A | Date B | A (MM-DD) | B (MM-DD) |
|---|---|---|---|---|
| CUST_248 | 2022-06-09 | 2022-09-06 | 06-09 | 09-06 |
| CUST_308 | 2018-01-03 | 2018-03-01 | 01-03 | 03-01 |
| CUST_333 | 2023-02-11 | 2023-11-02 | 02-11 | 11-02 |
| CUST_391 | 2023-03-11 | 2023-11-03 | 03-11 | 11-03 |
| CUST_466 | 2021-03-07 | 2021-07-03 | 03-07 | 07-03 |

**Root cause identified:** a DD/MM vs MM/DD ambiguity. This is a *parsing*
defect, not 5 typo events — so **neither** value in each pair can be trusted, and
the same ambiguity may affect other records that happen not to have a duplicate
exposing it.

**Rule:** keep first occurrence, raise `dq_dup_conflict`, exclude the 5 customers
from modelling. ⚠️ needs source review

#### DQ-03 · `customer_info.Age` — 35.00% missing
**Affected:** 3,500 of 10,000 customers

The missing rate is **exactly 35.0000%** and is flat across every segment:

| Segment | Missing rate |
|---|---|
| Female / Male | 0.3590 / 0.3409 |
| Rural / Suburban / Urban | 0.3470 / 0.3530 / 0.3499 |
| Monthly / Prepaid / Yearly | 0.3458 / 0.3541 / 0.3500 |
| Not churned / churned | 0.3495 / 0.3562 |

All sit inside ±3 standard errors (±0.0143) of the global rate, so the
missingness is **consistent with MCAR** — a clean random 35% drop, not
informative missingness driven by a subgroup.

**Rule:** do not drop rows (that would discard 35% of the base). Provide
`Age_imputed` (median 43) behind a flag column. ✅ auto-fixed with flag

#### DQ-04 · `customer_info.MonthlyCharges` — 49 values from a contaminant distribution
A two-component Gaussian mixture separates the charges into a clean body
(weight 0.9950, mean 49.97, sd 15.23) and a **contaminant (weight 0.0050, mean
257.42, sd 64.95)**, improving BIC from 89,797.6 → 83,645.6 (**ΔBIC = 6,152** —
decisive).

Supporting evidence:
- The highest charge is **390.30 — 7.8× the median** of 50.02
- A log-normal fitted to the clean body predicts **0.79 rows above 200**; **38 are observed**
- The density collapses by ~440× across the boundary, but the support stays
  continuous — the signature of values being *replaced* by draws from a wider
  distribution rather than of a genuine heavy tail

**Rule:** set `MonthlyCharges_clean = NULL`, raise `dq_charge_contaminated`. ✅ auto-fixed with flag

#### DQ-05 · `customer_info.MonthlyCharges` — 5 negative charges
Values: **−6.72, −4.55, −3.89, −3.15, −0.46**. A monthly charge cannot be negative.

**Rule:** set `MonthlyCharges_clean = NULL`, raise `dq_charge_negative`. ✅ auto-fixed with flag

#### DQ-07 · `usage_data` × `customer_info` — 2,278 rows dated before signup
**Affected:** 2,278 rows = 1.90% of usage, spanning 751 customers.

- median **48 days** before signup, maximum **276 days**
- 751 of the 920 customers who signed up in 2023 received activity from January 2023 onward

The dataset gives **every** customer a full 12-month 2023 history regardless of
when they joined. For customers who joined mid-2023, up to 11 months of activity
predates their own account.

A supporting oddity: signups run at a steady ~150/month from 2018-01 through
2023-06, then collapse to **1–3 per month** for 2023-07 → 2023-11 (11 customers
in total).

**Which side is wrong — the dates or the usage?** The 2023 cohort is otherwise
statistically indistinguishable from the rest (age 43.2 vs 43.4, calls 498.8 vs
491.8, churn 7.5% vs 7.9%), so the defect is confined to the date fields. I
cannot determine from the data alone whether `SignupDate` or the usage history
is authoritative. **This needs the data owner.**

**Rule:** flag via `dq_pre_signup`; exclude pre-signup rows from any
tenure-based metric. ⚠️ needs source review

---

### Severity: Medium

#### DQ-06 · `customer_info.MonthlyCharges` — 24 implausibly low charges
Range **0.38 – 8.74**, against a clean-body minimum of 8.89. Too small for any
tariff in this product set, but not logically impossible.

**Rule:** flag via `dq_charge_implausible_low` for review; not auto-nulled. ⚠️ review

#### DQ-09 · `usage_data.SMSCount` — no behavioural structure
The distribution is **exactly Uniform(0, 49)**:

| Statistic | Observed | Uniform(0,49) |
|---|---|---|
| Mean | 24.525 | 24.500 |
| Std dev | 14.424 | 14.430 |
| Kurtosis | **−1.200** | **−1.200** |
| Counts per value | 2,297 – 2,508 | 2,400 |

There is a hard stop at 49 (never 50). Real SMS behaviour is right-skewed with a
mode near zero and a long tail; this is flat noise. **This field carries no
signal and should not be used as a behavioural feature.**

**Rule:** not repairable. Record as a data-realism limitation. ℹ️ informational

#### DQ-10 · `usage_data` — `CallMinutes` and `DataUsageGB` are Gaussian synthetic draws
Kurtosis **0.189** and **−0.102** against **0.000** for a Normal. Real call and
data usage are strongly right-skewed (a few heavy users dominate volume); these
are symmetric.

**Rule:** not repairable. Record as a data-realism limitation. ℹ️ informational

---

### Severity: Low

#### DQ-08 · `usage_data.DataUsageGB` — 58 rows censored at exactly 0
A smooth Normal(9.90, 3.07) predicts ~0 rows sitting *exactly* at zero; 58 are
observed, with none below. Those rows average **518.6 call-minutes** versus
492.4 overall, so the customers were clearly active — the 0 is a recording
floor, not dormancy.

**Rule:** flag via `dq_data_zero_censored`; treat as censored in any average. ⚠️ review

---

## 6. Business impact of not cleaning

Joining the raw `customer_info` to `churn_labels` — the most natural first step
in any analysis — silently inflates every duplicated customer:

| Metric | Naive (raw) | Audited | Change |
|---|---|---|---|
| Rows after joining to `churn_labels` | 10,200 | 10,000 | **−200** |
| Churn rate | 7.8137% | 7.8600% | **+0.046 pp** |
| Mean `MonthlyCharges` | 50.962 | 50.001 | **−0.961** |
| Mean `Age` | 43.425 | 43.273 | −0.152 |

A 0.05 percentage-point error on the churn rate is small in absolute terms, but
it is **entirely an artefact**: it is not real movement in customer behaviour and
it is invisible unless someone checks row counts. The 0.96 shift in mean charges
is larger, and is contaminated by both the duplicate rows and the 78 invalid
charge values.

---

## 7. Limitations and uncertainty

Stated plainly, because these bound what the audit can claim:

- **No ground truth.** I inferred defect populations from statistical structure.
  DQ-02 and DQ-07 in particular need the data owner to confirm which value is
  authoritative — I can prove an inconsistency exists, not which side is wrong.
- **The 92-charge threshold on the right tail is a judgement call.** There is no
  empty gap in the distribution. The mixture model gives a principled cut at 49
  rows; a raw IQR fence gives 82. The 38 values above 200 are unambiguously
  wrong under any rule; the 92–200 band is a genuine judgement.
- **"No churn signal" is a strong negative claim.** It is well supported at this
  sample size (n = 10,000, 786 positives; three model families; a 20-draw
  permutation null), but it establishes only that *these* features fail to
  predict *this* label. It does not prove the labels are random.
- **Selection effects were not in scope** — there is no information about how
  these 10,000 customers were sampled from a population, so nothing here
  supports inference beyond the file.
- **The data is synthetic.** Gaussian/uniform usage metrics and a
  signal-free label point to generated data, so no finding should be read as a
  statement about real telecom customers.

---

## 8. Recommendations, in priority order

1. **Do not build a churn model on this data.** Report DQ-11 before anyone
   spends time tuning a classifier. If a model must be delivered, it needs a
   different label source.
2. **Escalate DQ-07 to the data owner.** Establish whether `SignupDate` or the
   usage history is authoritative. Until then, exclude the 2,278 pre-signup rows
   from tenure, LTV, and time-to-churn work.
3. **De-duplicate `customer_info` before any join.** This is a one-line fix and
   it silently corrupts every aggregate until it is applied.
4. **Review the DD/MM vs MM/DD ambiguity (DQ-02) at the ingestion layer**, not
   just for these 5 customers. A systematic parsing ambiguity will affect
   records that no duplicate happens to expose.
5. **Decide an explicit treatment for missing `Age`** — median imputation behind
   a flag is defensible under MCAR and preserves the full base; row deletion is
   not.
6. **Drop `SMSCount` from any behavioural feature set** (DQ-09) and avoid
   over-reading the symmetric usage distributions (DQ-10).
7. **Add the automated checks** in `scripts/` to the ingestion pipeline so these
   defects fail loudly next time rather than reaching an analyst.

---

## 9. Deliverables

```
outputs/
├── data_quality_register.csv        # 11 issues: rows, %, severity, evidence, rule
├── impact_naive_vs_audited.csv      # naive-vs-cleaned metric comparison
├── audit_summary.json               # verified headline numbers
├── clean/
│   ├── customer_info_clean.csv      # 10,000 rows, duplicate-free, flagged
│   ├── usage_data_clean.csv         # 120,000 rows, pre-signup rows flagged
│   └── churn_labels_clean.csv       # unchanged
└── figures/
    ├── fig1_monthly_charges.png     # the price contaminant, isolated by mixture
    ├── fig2_missingness_and_sanity.png
    ├── fig3_usage_shapes.png        # uniform/Gaussian synthetic signatures
    ├── fig4_churn_signal.png        # models vs shuffled-label null band
    └── fig5_signup_timeline.png
scripts/01_profile.py … verify_readme.py  # the full audit, re-runnable
```

**Reproduce:** `uv venv .venv && uv pip install --python .venv pandas numpy matplotlib seaborn scipy scikit-learn pillow`
then run `scripts/01` → `08` in order with `.venv/Scripts/python.exe`, and finish with
`scripts/verify_readme.py` to confirm the documented figures still match the outputs.

---

## Document navigation

| | |
|---|---|
| **Previous** | — (this is the second of two documents) |
| **Up** | [Project README — the case study](../README.md) |
| **Next** | — |

**Back to:** [Navigation](#navigation) · [Top of document](#data-quality-audit--telecom-customer-churn-dataset) · [Project README](../README.md)
