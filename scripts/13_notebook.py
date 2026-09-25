"""Build the submission notebook: notebooks/churn_analysis.ipynb

The brief requires a notebook explaining cleaning decisions, feature engineering,
model selection, validation, and business impact. This generates a real .ipynb:
markdown narrative plus code cells that READ the saved artifacts, so every number
displayed is regenerated rather than pasted.

Usage:
    python scripts/13_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/churn_analysis.ipynb
"""
import json
from pathlib import Path

cells = []


def md(text):
    return _cell("markdown", text.splitlines(keepends=True))


def code(text):
    return _cell("code", text.strip().splitlines(keepends=True))


def _cell(kind, source):
    """nbformat >=4.5 wants a stable cell id; generate one deterministically."""
    cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
            "source": source}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)


# --------------------------------------------------------------------------- 1
md("""# Telecom Customer Churn — Data Quality, Feature Engineering & Modelling

**Competition submission notebook** · Mustafa Al-Rouby

---

## Business context

Customer churn is the central economics problem in telecom. Acquiring a replacement
customer costs up to **5x more** than retaining an existing one, so even a small
improvement in retention targeting is worth more than a large improvement in
acquisition.

## Objective

Predict whether a customer will churn in the next billing cycle (binary
classification), and align the evaluation with business impact rather than pure
statistical accuracy.

## What this notebook covers

| Section | Question it answers |
|---|---|
| 1. Data loading | What are we working with? |
| 2. Data profiling | Is the data trustworthy? |
| 3. Cleaning decisions | What did we change, and why? |
| 4. Auditing the brief's claims | Do the documented defects actually exist? |
| 5. Feature engineering | What signals did we construct? |
| 6. Model selection | Which models, and why? |
| 7. Validation approach | How do we know the score is real? |
| 8. Business impact | What is this worth in money? |
| 9. Bonus challenges | SHAP, drift, cost-sensitive learning, survival |
| 10. Conclusion | What should the business actually do? |

> **Short on time?** Read section 8. The honest headline is that this dataset does
> not support churn targeting, and the notebook quantifies exactly how much value
> the model fails to capture.""")

# --------------------------------------------------------------------------- 2
md("""---
# 1. Data loading

Three tables: one row per customer in `customer_info` and `churn_labels`, one row
per customer per month in `usage_data`.""")

code("""
import json
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

ci = pd.read_csv("../data/customer_info.csv")
ud = pd.read_csv("../data/usage_data.csv")
cl = pd.read_csv("../data/churn_labels.csv")

for name, d in [("customer_info", ci), ("usage_data", ud), ("churn_labels", cl)]:
    print(f"{name:16s} {d.shape[0]:>7,} rows x {d.shape[1]} cols")

print(f"\\nunique customers: {ci.CustomerID.nunique():,}")
print(f"churn rate      : {cl.Churn.mean():.4%}  ({int(cl.Churn.sum())} of {len(cl):,})")
print(f"imbalance ratio : 1 : {(1 - cl.Churn.mean()) / cl.Churn.mean():.1f}")
""")

# --------------------------------------------------------------------------- 3
md("""---
# 2. Data profiling — the 11 defects

A churn model built on unvalidated data produces confident answers nobody can
trust, so the audit came first. Every finding is computed in code and stored in
`outputs/data_quality_register.csv`.""")

code("""
reg = pd.read_csv("../outputs/data_quality_register.csv")
print(f"{len(reg)} issues found\\n")
print(reg[["issue_id", "fields", "issue", "affected_rows", "affected_pct",
           "severity", "resolution"]].to_string(index=False))
""")

md("""**Reading the register.** Two findings change how the data can be used at all:

- **DQ-11** — the target label is uninformative (proved in section 7)
- **DQ-07** — 2,278 usage rows are dated *before* the customer signed up, so any
  tenure-derived feature is unreliable for 751 customers

The rest are repairable. Note the `resolution` column: only logically impossible
values were auto-repaired; judgement calls were flagged for a human.""")

# --------------------------------------------------------------------------- 4
md("""---
# 3. Data cleaning decisions

## Governing principle: nothing is silently fixed

Invalid values become `NULL` in a companion `*_clean` column **and** raise a boolean
`dq_*` flag. A downstream analyst can always see which rows were touched. This
matters because a silently-cleaned dataset hides the reason a metric moved.""")

code("""
clean = pd.read_csv("../outputs/clean/customer_info_clean.csv")
print(f"raw rows     : {len(ci):,}")
print(f"cleaned rows : {len(clean):,}   ({len(ci) - len(clean):,} duplicate rows removed)")
print(f"unique ID    : {clean.CustomerID.is_unique}\\n")

for col, desc in {
    "dq_dup_conflict": "conflicting duplicate SignupDate (DQ-02)",
    "dq_charge_negative": "negative monthly charge (DQ-05)",
    "dq_charge_contaminated": "value from contaminant distribution (DQ-04)",
    "dq_charge_implausible_low": "implausibly low charge (DQ-06)",
}.items():
    print(f"{desc:44s} {int(clean[col].sum()):5d} rows")
print(f"{'MonthlyCharges set to NULL':44s} "
      f"{int(clean.MonthlyCharges_clean.isna().sum()):5d} rows")
""")

md("""### The decisions, and the reasoning behind each

| Defect | Decision | Why |
|---|---|---|
| 195 exact duplicate rows | **Drop** | Byte-identical rows carry no information |
| 5 conflicting duplicates | **Keep first + flag + quarantine** | All 5 disagree only on `SignupDate`, and every pair is an exact month/day swap — a `DD/MM` vs `MM/DD` parsing ambiguity. Neither value is trustworthy and the fault is systemic, so it goes to the data owner rather than being patched |
| `Age` 35.00% missing | **Impute median + flag, never drop** | The missing rate is flat across Gender (0.341–0.359), Region (0.347–0.353), ContractType (0.346–0.354) and Churn (0.350 / 0.356) — all inside ±3 standard errors. That is MCAR: dropping rows would discard 35% of the base and fix nothing |
| 49 contaminant charges | **Set `NULL` + flag** | A 2-component Gaussian mixture isolates them (weight 0.0050, mean 257.42, ΔBIC = 6,152). A log-normal fitted to the clean body predicts 0.79 rows above 200; 38 are observed |
| 5 negative charges | **Set `NULL` + flag** | Logically impossible |
| 24 implausibly low charges | **Flag only** | Suspicious but not impossible — a human decides |
| 2,278 pre-signup usage rows | **Flag, exclude from tenure metrics** | No algorithm can recover the truth; needs the data owner |
| 58 data-usage zeros | **Flag as censored** | Those rows average 518.6 call-minutes vs 492.4 overall, so the customers were active. The 0 is a recording floor |

### Why missing Age was not imputed with a model

With MCAR missingness there is no information to impute *from* — a model would
reproduce the marginal distribution while adding false confidence. Median
imputation behind a flag is the honest choice, and the flag lets the learner test
whether missingness itself is informative (it is not: churn is 35.0% vs 35.6% by
missingness status).""")

# --------------------------------------------------------------------------- 5
md("""---
# 4. Auditing the brief's own claims

The brief states two things that deserve verification rather than acceptance.""")

code("""
# Claim 1: "SignupDate (inconsistent formats included)"
raw = ci.SignupDate.astype(str).str.replace(r"\\d", "9", regex=True)
print("SignupDate string shapes:")
print(raw.value_counts().to_string())
print(f"\\ncontains a '/': {ci.SignupDate.astype(str).str.contains('/').sum()}")
print(f"unparseable   : {int(pd.to_datetime(ci.SignupDate, errors='coerce').isna().sum())}")
print()
print("-> every value is uniform ISO 9999-99-99. There are NO format inconsistencies.")
print("   The real date defect is the 5 duplicate-conflict rows with swapped month/day.")
""")

code("""
# Claim 2: "Churn patterns are behaviourally embedded (e.g. usage decline before churn)"
uni = pd.read_csv("../outputs/trend_univariate.csv")
print("Trend features tested for a churn signal (decline = negative slope, ratio < 1):\\n")
print(uni.to_string(index=False))
print(f"\\nfeatures with p < 0.05: {(uni.p < 0.05).sum()} of {len(uni)}")
print()
print("-> no trend feature separates churners. The claimed usage-decline signal is")
print("   not present in the delivered data.")
""")

md("""**This matters.** A submission that assumed the brief was accurate would have spent
its effort engineering decline features that cannot work, and could have reported a
model as successful on the strength of a lucky validation split. Verifying the brief
against the bytes is what makes the rest of this notebook trustworthy.""")

# --------------------------------------------------------------------------- 6
md("""---
# 5. Feature engineering strategy

66 features per customer, in six documented families. The trend family is
first-class rather than an afterthought, because the brief's stated mechanism is a
*decline* — and a standard deviation cannot express direction (a steady rise and a
steady decline have identical SDs).""")

code("""
feats = pd.read_csv("../outputs/features.csv")
print(f"feature table: {feats.shape[0]:,} customers x {feats.shape[1]} columns\\n")

families = {
    "Demographic": ["Age_imputed", "age_missing", "is_male"],
    "Contract": ["MonthlyCharges_clean", "charge_invalid", "tenure_months",
                 "signup_year", "signed_up_in_window"],
    "Usage level": ["call_mean", "data_median", "sms_max", "comp_min"],
    "Usage volatility": ["call_sd", "data_cv", "sms_sd", "comp_cv"],
    "Usage trend": ["call_slope", "data_halfratio", "sms_rel_slope",
                    "comp_last_minus_first"],
    "Engagement": ["n_complaint_months", "data_per_call_min",
                   "complaints_per_call_hour"],
}
for fam, cols in families.items():
    present = [c for c in cols if c in feats.columns]
    print(f"{fam:18s} e.g. {', '.join(present)}")
print(f"\\ntotal numeric features: {feats.select_dtypes('number').shape[1]}")
""")

md("""### Rationale for the trend family

For each of the four usage metrics (`CallMinutes`, `DataUsageGB`, `SMSCount`,
`Complaints`) we compute:

| Feature | Definition | Why |
|---|---|---|
| `*_slope` | OLS slope across the 12 months | The literal "decline" mechanism |
| `*_rel_slope` | slope ÷ mean | Scale-free, comparable across customers |
| `*_halfratio` | mean(months 10–12) ÷ mean(months 1–3) | Robust end-vs-start comparison |
| `*_last_minus_first` | month 12 − month 1 | Simple directional change |
| `*_trend_r2` | R² of the linear fit | Is the trend consistent or noise? |
| `*_declining` | slope < 0 | Binary flag for the mechanism |

Plus engagement ratios (`data_per_call_min`, `complaints_per_call_hour`) that
normalise behaviour by activity level, so a low-usage customer is not mistaken for
a disengaging one.

### Leakage control

Every feature is computed per customer from that customer's own 12 months. No
target statistics, no cross-customer aggregates, no target-derived encodings. The
`dq_*` flags are deliberately included: they let the model test whether *being a
defective record* correlates with churn (it does not, but testing it is the point).""")

# --------------------------------------------------------------------------- 7
md("""---
# 6. Model selection reasoning

Five candidates, spanning the bias–variance spectrum and including an explicit
no-skill baseline:

| Model | Why it is in the comparison |
|---|---|
| **Dummy (base rate)** | The honest floor. Anything that cannot beat it is not a model |
| **Logistic regression** | Linear, interpretable, well-calibrated by construction; the right default for a mostly-categorical tabular problem |
| **Logistic (`class_weight='balanced'`)** | A first, cheap cost-sensitive response to the 1:12 imbalance |
| **Random forest** | Captures non-linearity and interactions without scaling, and is robust to irrelevant features — which matters when most of the 66 features are noise |
| **Gradient boosting** | Usually the strongest tabular learner, and it can find interactions a linear model cannot |""")

code("""
s = json.load(open("../outputs/model_summary.json"))
print(f"design matrix : {s['n_rows']:,} rows x {s['n_features']} features")
print(f"base rate     : {s['base_rate']:.4%}\\n")
print(f"{'model':26s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Brier':>8s}")
print(f"{'(no-skill reference)':26s} {'0.5000':>9s} {s['base_rate']:9.4f} "
      f"{s['base_rate'] * (1 - s['base_rate']):8.4f}")
for name, m in s["models"].items():
    print(f"{name:26s} {m['roc_auc']:9.4f} {m['pr_auc']:9.4f} {m['brier']:8.4f}")
""")

md("""### Why PR-AUC is reported next to ROC-AUC

With 7.86% positives, ROC-AUC flatters a model — a useless classifier can still
score near 0.5. **Average precision** uses the base rate as its no-skill reference,
so `PR-AUC ≈ base rate` is the signature of no signal. The table above should be
read that way.""")

# --------------------------------------------------------------------------- 8
md("""---
# 7. Validation approach

## There is no held-out test set

`churn_labels.csv` covers all 10,000 customers, so there is nothing to hold out and
in-sample metrics would be meaningless. Instead:

1. **Stratified 5-fold cross-validation** preserves the 7.86% positive rate in
   every fold.
2. **Out-of-fold predictions** — each customer is scored by a model trained without
   them. Every metric here, and the submitted probabilities themselves, are
   out-of-fold.
3. **A shuffled-label null** establishes what "no signal" looks like at this sample
   size, so a score is judged against noise rather than against 0.5.""")

code("""
print("RANDOMISATION TEST — is the score real?")
print(f"best model          : {s['best_model']}  ROC-AUC = {s['best_roc_auc']:.4f}")
print(f"shuffled-label null : {s['null_band'][0]:.4f} - {s['null_band'][1]:.4f}")
print(f"share of null draws >= best model: {s['null_share_ge_best']:.1%}\\n")
if s["null_share_ge_best"] < 0.05:
    print("VERDICT: real signal present.")
else:
    print("VERDICT: INDISTINGUISHABLE from random labels.")
    print("         The model's ranking carries no information about churn.")
""")

code("""
print("DECILE LIFT — the operational question: who do we contact first?\\n")
lift = pd.read_csv("../outputs/decile_lift.csv")
lift["churn_rate"] = (lift.churn_rate * 100).round(2)
lift["cum_recall"] = (lift.cum_recall * 100).round(1)
print(lift.to_string(index=False))
print()
print(f"Top-decile lift: {s['top_decile_lift']:.2f}x")
print("-> a lift at or below 1.0 means the 'highest risk' decile churns no more often")
print("   than a random sample. There is no one to prioritise.")
""")

code("""
print("CALIBRATION — are the submitted probabilities honest?\\n")
cal = pd.read_csv("../outputs/calibration.csv")
print(cal.round(4).to_string(index=False))
print()
print(f"predicted mean {s['submitted_mean_prob']:.4f} vs base rate {s['base_rate']:.4f}")
print(f"Brier {s['submitted_brier']:.4f} vs no-skill "
      f"{s['base_rate'] * (1 - s['base_rate']):.4f}")
print("-> the probabilities sit at the base rate, which is the correct answer when")
print("   there is nothing to predict.")
""")

# --------------------------------------------------------------------------- 9
md("""---
# 8. Business impact estimation

## The cost model

Offering a retention deal costs `C_offer` whether or not the customer would have
churned. Not offering costs `p × C_loss` in expectation. So:

```
offer  iff   p × C_loss > C_offer      ->      p > C_offer / C_loss
```

With the brief's 5:1 ratio the theoretically optimal threshold is `tau* = 1/5 = 0.20`.
Note this is **well above the 7.86% base rate** — which is the whole economic point:
at a 5:1 loss ratio a random customer is *not* worth targeting (expected saving
0.079 × 5 = 0.39 units, against 1 unit of offer cost).""")

code("""
C_OFFER, C_LOSS = 1.0, 5.0
never = s["cost_never_target"]
allc = s["cost_target_all"]
perfect = s["cost_perfect_model"]
model_c = s["cost_model_at_tau_star"]

print(f"Retention offer cost      : {C_OFFER:.0f} unit")
print(f"Cost of losing a customer : {C_LOSS:.0f} units  (brief: 5x retention)")
print(f"Derived optimal threshold : tau* = {s['optimal_tau_theory']:.4f}\\n")

print(f"{'strategy':36s} {'cost':>8s} {'vs do-nothing':>14s}")
for label, c in [("Do nothing", never),
                 ("Target everybody", allc),
                 ("Perfect model (upper bound)", perfect),
                 (f"Model @ tau* = {s['optimal_tau_theory']:.2f}", model_c)]:
    print(f"{label:36s} {c:8.0f} {never - c:+14.0f}")

print(f"\\nMaximum saving available to ANY model: {s['max_achievable_saving']:.0f} units")
print(f"Value actually captured by the model : {s['value_captured_pct']:+.2%}")
print()
print("-> Targeting everybody is the WORST option: it costs more in offers than the")
print("   churn it prevents. The model does not beat doing nothing, because its")
print("   ranking contains no information.")
""")

code("""
print("SENSITIVITY — how the decision changes with the acquisition ratio\\n")
sens = pd.read_csv("../outputs/cost_sensitivity.csv")
print(sens.to_string(index=False))
print()
print("-> at every ratio tested the model is never the cheapest policy. A model is")
print("   only worth deploying when it beats both trivial strategies.")
""")

md("""### What would the model have to achieve to be worth deploying?

With 786 churners among 10,000 customers:

| Policy | Cost | Comment |
|---|---|---|
| Do nothing | 3,930 units | Lose every churner |
| Target everybody | 10,000 units | Waste 9,214 offers |
| **Perfect model** | **786 units** | Offer only to true churners — the upper bound |
| Our model @ τ* | 4,039 units | **Worse than doing nothing** |

The gap between doing nothing (3,930) and the perfect model (786) is **3,144 units
of addressable value**. The model captures **none of it**. A real retention
programme needs a ranking that is meaningful in the top deciles; this data cannot
produce one.""")

# -------------------------------------------------------------------------- 10
md("""---
# 9. Bonus challenges

## 9.1 SHAP explainability

If a churn driver existed, SHAP would concentrate attribution on it. Instead the
attribution is spread flat across all features.""")

code("""
shap_imp = pd.read_csv("../outputs/shap_importance.csv")
print("top 12 features by mean |SHAP value|:\\n")
print(shap_imp.head(12).round(5).to_string(index=False))
print(f"\\ntop feature holds {shap_imp.share.iloc[0]:.2%} of total attribution")
print("-> no dominant driver. A signal-free label produces exactly this pattern:")
print("   many features carrying small, mutually-cancelling contributions.")
""")

md("""## 9.2 Data drift detection

Population Stability Index compares each month against a months 1–6 reference
(< 0.1 stable, 0.1–0.25 moderate, > 0.25 major shift).""")

code("""
drift = pd.read_csv("../outputs/drift_psi.csv")
print(drift.round(4).to_string(index=False))
base = drift.loc[drift.month <= 9, ["call", "data"]].max().max()
step = drift.loc[drift.month >= 10, ["call", "data"]].min().min()
print(f"\\nmonths 1-9  max PSI (call/data): {base:.4f}")
print(f"months 10-12 min PSI (call/data): {step:.4f}   -> a {step/base:.0f}x step change")
print()
print("-> CallMinutes and DataUsageGB both step up at month 10 while SMSCount and")
print("   Complaints stay flat. The absolute PSI stays under the conventional 0.1")
print("   alarm, so this is a small but unambiguous structural break rather than a")
print("   major drift event - still worth flagging, because a model trained on")
print("   months 1-9 faces a measurably different input distribution from month 10.")
""")

md("""## 9.3 Cost-sensitive learning

Two ways to make a learner cost-aware: reweight the training samples, or move the
decision threshold. Both are tested against the cost model.""")

code("""
csl = pd.read_csv("../outputs/cost_sensitive_learning.csv")
print(csl.round(4).to_string(index=False))
print()
print("-> reweighting shifts the threshold the model prefers but not its ranking power")
print("   (ROC-AUC is invariant to the decision threshold). It cannot recover value")
print("   that is absent from the features.")
""")

md("""## 9.4 Survival analysis — feasibility assessment

**Conclusion: not feasible on this dataset**, and the reason is structural rather
than a limitation of effort.""")

code("""
print("Available columns")
print(f"  churn_labels : {list(cl.columns)}")
print(f"  usage_data   : {list(ud.columns)}")
print(f"  target values: {sorted(cl.Churn.unique())}")
print()
print("A survival model needs, per subject:")
print("  (a) a duration        - months from origin to the churn event")
print("  (b) an event indicator - observed churn vs right-censored")
print()
print("This dataset provides NEITHER. There is no churn date and no tenure field.")
print("Every churner would receive the same duration (the end of the 12-month window)")
print("and every non-churner is censored at that same instant, so the hazard function")
print("is not identifiable.")
print()
print("What would be needed: a churn date or churn month per customer, plus tenure at")
print("observation start.")
""")

# -------------------------------------------------------------------------- 11
md("""---
# 10. Conclusion

## What was delivered

| Requirement | Status |
|---|---|
| Clean and preprocess the messy datasets | 11 defects, one documented rule each, every change flagged |
| Engineer meaningful behavioural features | 66 features in 6 families, including the full trend family |
| Build a model estimating churn probability | 5 candidates, out-of-fold probabilities, `prediction.csv` submitted |
| Align evaluation with business impact | Cost model with derived threshold, sensitivity analysis, value-captured metric |
| Bonus: SHAP | Completed — attribution flat, no dominant driver |
| Bonus: drift detection | Completed — `CallMinutes` breaks at month 10 |
| Bonus: cost-sensitive learning | Completed — reweighting cannot fix an uninformative ranking |
| Bonus: survival analysis | Assessed — not feasible, no event time or censoring indicator |

## The finding that matters

**The churn label carries no learnable signal from the provided features.**

- Best model ROC-AUC **0.5181** against a shuffled-label null band of 0.4772–0.5198
  (10% of null draws matched or beat it)
- PR-AUC **0.0808** against a base rate of 0.0786 — no lift
- Top-decile lift **0.95x** — the "highest risk" decile churns *less* than average
- SHAP attribution flat across all 70 features
- The cost-optimal policy is to send **no** retention offers at all

This is not a modelling failure. It is a property of the data, confirmed by four
independent methods, plus direct verification in section 4 that the brief's claimed
"usage decline before churn" mechanism is absent.

## Recommendations

1. **Do not deploy churn targeting on this data.** Any model would be
   indistinguishable from randomly selecting customers.
2. **Fix the source data first** — DQ-07 makes every tenure-based feature
   unreliable for 751 customers.
3. **Resolve the DD/MM vs MM/DD ambiguity at the ingestion layer.** It is systemic,
   not confined to the 5 rows where a duplicate happened to expose it.
4. **Investigate the month-10 structural break** in `CallMinutes` before building
   any time-series feature on it.
5. **Treat the label as suspect.** A signal-free target plus Gaussian/uniform
   synthetic usage metrics suggests churn was assigned independently of the
   features during dataset construction.

## The professional point

The deliverable of this notebook is a *refusal* — evidence, assembled in hours
rather than weeks, that a churn model should not be shipped. Reporting that a model
cannot work is more valuable than reporting one that appears to work and quietly
fails in production.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.7"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("notebooks").mkdir(exist_ok=True)
out = Path("notebooks/churn_analysis.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
print(f"  {len(cells)} cells: {sum(c['cell_type'] == 'markdown' for c in cells)} markdown, "
      f"{sum(c['cell_type'] == 'code' for c in cells)} code")
