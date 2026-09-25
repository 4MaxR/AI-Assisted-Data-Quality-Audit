"""Pass 7: repair pipeline.

Applies one documented rule per defect, tags every change so it is auditable,
writes cleaned tables + a machine-readable issue register, and measures the
before/after impact on headline metrics.

Design choice: nothing is silently fixed. Invalid values become NULL with a
companion flag column, so a downstream analyst can always see what was touched.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

RAW = "data"
OUT = "outputs"
CLEAN = f"{OUT}/clean"

import os
os.makedirs(CLEAN, exist_ok=True)

ci_raw = pd.read_csv(f"{RAW}/customer_info.csv")
ud_raw = pd.read_csv(f"{RAW}/usage_data.csv")
cl = pd.read_csv(f"{RAW}/churn_labels.csv")

register = []


def log_issue(**kw):
    register.append(kw)


# ---------------------------------------------------------------- DQ-01 / DQ-02
n_raw = len(ci_raw)
n_ids = ci_raw.CustomerID.nunique()
dup_mask = ci_raw.CustomerID.duplicated(keep=False)
dup_ids = ci_raw.loc[dup_mask, "CustomerID"].unique()
exact_ids, conflict_ids = [], []
for cid in dup_ids:
    g = ci_raw[ci_raw.CustomerID == cid]
    (exact_ids if len(g.drop_duplicates()) == 1 else conflict_ids).append(cid)

# rule: exact duplicates -> drop (identical rows carry no information)
#       conflicting duplicates -> keep first, flag, quarantine from modelling
ci = ci_raw.drop_duplicates(subset="CustomerID", keep="first").copy()
ci["dq_dup_conflict"] = ci.CustomerID.isin(conflict_ids)

log_issue(issue_id="DQ-01", table="customer_info", fields="all",
          issue="Exact duplicate customer records",
          affected_rows=len(exact_ids), affected_pct=round(len(exact_ids) / n_ids * 100, 2),
          severity="High",
          evidence=f"{n_raw:,} rows for {n_ids:,} unique CustomerIDs; "
                   f"{len(exact_ids)} ID groups byte-identical",
          rule="Drop duplicate rows (keep first occurrence)",
          resolution="auto-fixed")
log_issue(issue_id="DQ-02", table="customer_info", fields="SignupDate",
          issue="Conflicting duplicate records (same ID, different SignupDate)",
          affected_rows=len(conflict_ids) * 2,
          affected_pct=round(len(conflict_ids) * 2 / n_raw * 100, 2),
          severity="High",
          evidence=f"IDs {', '.join(conflict_ids)} each appear twice with every other "
                   f"field identical and only SignupDate differing - and in all 5 cases "
                   f"the two dates are an exact month/day swap (CUST_248: 2022-06-09 vs "
                   f"2022-09-06; CUST_308: 2018-01-03 vs 2018-03-01; CUST_333: 2023-02-11 "
                   f"vs 2023-11-02; CUST_391: 2023-03-11 vs 2023-11-03; CUST_466: "
                   f"2021-03-07 vs 2021-07-03), i.e. a DD/MM vs MM/DD parsing ambiguity. "
                   f"Neither value in each pair can be trusted, and the same ambiguity "
                   f"may affect records that no duplicate happens to expose",
          rule="Keep first occurrence, set dq_dup_conflict flag, exclude from modelling",
          resolution="flagged-for-review")

# ---------------------------------------------------------------------- DQ-03
age_missing = int(ci.Age.isna().sum())
ci["Age_imputed"] = ci.Age.fillna(ci.Age.median())
log_issue(issue_id="DQ-03", table="customer_info", fields="Age",
          issue="Missing values (missing completely at random)",
          affected_rows=age_missing, affected_pct=round(age_missing / n_ids * 100, 2),
          severity="High",
          evidence="Exactly 3,500/10,000 = 35.0000% missing; missing rate is flat "
                   "across Gender (0.341-0.359), Region (0.347-0.353), ContractType "
                   "(0.346-0.354) and Churn (0.350/0.356) - all inside 3*SE sampling noise",
          rule="Do not drop rows. Provide Age_imputed (median = "
               f"{ci.Age.median():.0f}) behind an explicit flag column",
          resolution="auto-fixed-with-flag")

# ---------------------------------------------------------------- DQ-04/05/06
mc = ci.MonthlyCharges.copy()
X = mc.values.reshape(-1, 1)
gmm = GaussianMixture(2, random_state=0, n_init=5).fit(X)
means = gmm.means_.ravel()
contam = int(np.argmax(means))          # the high-mean component
post = gmm.predict_proba(X)[:, contam]
contaminated = post > 0.5
negatives = mc < 0
tiny = (mc >= 0) & (mc < 8.75)

ci["dq_charge_contaminated"] = contaminated
ci["dq_charge_negative"] = negatives
ci["dq_charge_implausible_low"] = tiny
ci["MonthlyCharges_clean"] = mc.where(~(contaminated | negatives), np.nan)

log_issue(issue_id="DQ-04", table="customer_info", fields="MonthlyCharges",
          issue="Values from a distinct contaminant distribution (implausible spend)",
          affected_rows=int(contaminated.sum()),
          affected_pct=round(contaminated.mean() * 100, 2),
          severity="High",
          evidence=f"2-component Gaussian mixture separates a component with weight "
                   f"{gmm.weights_[contam]:.4f} (mean {means[contam]:.0f}, sd "
                   f"{gmm.covariances_[contam,0,0]**0.5:.0f}); BIC improves 89,798 -> 83,646 "
                   f"(dBIC=6,152). A log-normal fitted to the clean body predicts 0.79 "
                   f"rows above 200 but 38 are observed. Highest value 390.30 = 7.8x the median",
          rule="Set MonthlyCharges_clean = NULL and flag via dq_charge_contaminated",
          resolution="auto-fixed-with-flag")
log_issue(issue_id="DQ-05", table="customer_info", fields="MonthlyCharges",
          issue="Negative monthly charges (logically impossible)",
          affected_rows=int(negatives.sum()), affected_pct=round(negatives.mean() * 100, 2),
          severity="High",
          evidence=f"Values: {', '.join(f'{v:.2f}' for v in sorted(mc[negatives]))}",
          rule="Set MonthlyCharges_clean = NULL and flag via dq_charge_negative",
          resolution="auto-fixed-with-flag")
log_issue(issue_id="DQ-06", table="customer_info", fields="MonthlyCharges",
          issue="Implausibly low positive charges (below the clean-body minimum)",
          affected_rows=int(tiny.sum()), affected_pct=round(tiny.mean() * 100, 2),
          severity="Medium",
          evidence=f"Range {mc[tiny].min():.2f}-{mc[tiny].max():.2f} against a clean-body "
                   f"minimum of 8.89 - too small for any tariff in this product set",
          rule="Flag via dq_charge_implausible_low for review (not auto-nulled)",
          resolution="flagged-for-review")

# ---------------------------------------------------------------------- DQ-07
ci["SignupDate_dt"] = pd.to_datetime(ci.SignupDate)
ud = ud_raw.copy()
ud["Month_dt"] = pd.to_datetime(ud.Month)
ud = ud.merge(ci[["CustomerID", "SignupDate_dt", "dq_dup_conflict"]], on="CustomerID")
ud["dq_pre_signup"] = ud.Month_dt < ud.SignupDate_dt
n_pre = int(ud.dq_pre_signup.sum())
n_pre_cust = ud.loc[ud.dq_pre_signup, "CustomerID"].nunique()
lag = (ud.loc[ud.dq_pre_signup, "SignupDate_dt"] - ud.loc[ud.dq_pre_signup, "Month_dt"]).dt.days
y2023 = int((ci.SignupDate_dt.dt.year == 2023).sum())
log_issue(issue_id="DQ-07", table="usage_data x customer_info",
          fields="Month vs SignupDate",
          issue="Usage rows dated before the customer's signup date",
          affected_rows=n_pre,
          affected_pct=round(n_pre / len(ud) * 100, 2),
          severity="High",
          evidence=f"{n_pre:,} rows ({n_pre/len(ud)*100:.2f}%) across {n_pre_cust:,} customers; "
                   f"median {lag.median():.0f} days and up to {lag.max()} days before signup. "
                   f"{n_pre_cust} of the {y2023} customers who signed up in 2023 received "
                   f"activity from Jan 2023 - the dataset hands every customer a full "
                   f"12-month 2023 history regardless of when they joined",
          rule="Cannot be repaired without the source of truth. Flag via dq_pre_signup; "
               "exclude pre-signup rows when computing tenure-based metrics",
          resolution="flagged-for-review")

# ---------------------------------------------------------------------- DQ-08
zero_data = ud.DataUsageGB == 0
mean_calls_zero = ud.loc[zero_data, "CallMinutes"].mean()
mean_calls_all = ud.CallMinutes.mean()
ud["dq_data_zero_censored"] = zero_data
log_issue(issue_id="DQ-08", table="usage_data", fields="DataUsageGB",
          issue="Point mass at exactly 0 GB while call/SMS activity is normal "
                "(left-censoring at a floor, not genuine zero usage)",
          affected_rows=int(zero_data.sum()), affected_pct=round(zero_data.mean() * 100, 3),
          severity="Low",
          evidence=f"A smooth Normal(9.90, 3.07) predicts ~0 rows exactly at zero; "
                   f"{int(zero_data.sum())} are observed with none below. Those rows average "
                   f"{mean_calls_zero:.1f} call-minutes vs {mean_calls_all:.1f} overall, so the "
                   f"customers were clearly active",
          rule="Flag via dq_data_zero_censored; treat as censored in any average",
          resolution="flagged-for-review")

# ------------------------------------------------------------------ DQ-09/11
sms = ud.SMSCount
kurt_sms = float(stats.kurtosis(sms, fisher=True))
call_kurt = float(stats.kurtosis(ud.CallMinutes, fisher=True))
data_kurt = float(stats.kurtosis(ud.DataUsageGB, fisher=True))
log_issue(issue_id="DQ-09", table="usage_data", fields="SMSCount",
          issue="Distribution is exactly Uniform(0,49) - carries no behavioural structure",
          affected_rows=len(ud), affected_pct=100.0, severity="Medium",
          evidence=f"mean 24.525 vs uniform 24.500; sd 14.424 vs uniform 14.430; "
                   f"kurtosis {kurt_sms:.3f} vs uniform -1.200; value counts flat "
                   f"(2,297-2,508 per value) with a hard stop at 49. Real SMS behaviour "
                   f"is right-skewed with a mode near zero",
          rule="Not repairable - record as a data-realism limitation",
          resolution="informational")
log_issue(issue_id="DQ-10", table="usage_data", fields="CallMinutes, DataUsageGB",
          issue="Behavioural metrics are Gaussian synthetic draws, not real usage shapes",
          affected_rows=len(ud), affected_pct=100.0, severity="Medium",
          evidence=f"Kurtosis {call_kurt:.3f} (CallMinutes) and {data_kurt:.3f} (DataUsageGB) "
                   f"against 0.000 for a Normal. Real call and data usage are strongly "
                   f"right-skewed; these are symmetric",
          rule="Not repairable - record as a data-realism limitation",
          resolution="informational")

# ---------------------------------------------------------------------- DQ-11
feat = ud.groupby("CustomerID").agg(**{
    "call_min": ("CallMinutes", "mean"), "call_max": ("CallMinutes", "max"),
    "call_sd": ("CallMinutes", "std"), "data_gb": ("DataUsageGB", "mean"),
    "data_sd": ("DataUsageGB", "std"), "sms": ("SMSCount", "mean"),
    "comp_sum": ("Complaints", "sum"), "comp_max": ("Complaints", "max"),
    "n_zero_data": ("DataUsageGB", lambda s: (s == 0).sum()),
})
log_issue(issue_id="DQ-11", table="churn_labels", fields="Churn",
          issue="The churn label has no detectable relationship with any provided feature "
                "- the dataset cannot support a predictive churn model",
          affected_rows=len(cl), affected_pct=100.0, severity="Critical (use case)",
          evidence="5-fold CV ROC-AUC 0.5176 (logistic), 0.5170 (random forest, 500 trees), "
                   "0.5164 (gradient boosting) against a 0.5000 random baseline. Shuffling the "
                   "labels 20 times produced null AUCs up to 0.5372, i.e. the real models score "
                   "inside the noise band. Single-feature tests: all p>0.11 except SMSCount "
                   "(p=0.026), which is 1 of 10 tests and therefore expected by chance",
          rule="Do not sell or build a predictive model on this data; report the label as "
               "an uninformative target and treat the exercise as coverage/quality only",
          resolution="informational")

# ---------------------------------------------------------------------- export
ci_out = ci.drop(columns=["SignupDate_dt"])
ci_out.to_csv(f"{CLEAN}/customer_info_clean.csv", index=False)
ud_out = ud.drop(columns=["Month_dt", "SignupDate_dt"])
ud_out.to_csv(f"{CLEAN}/usage_data_clean.csv", index=False)
cl.to_csv(f"{CLEAN}/churn_labels_clean.csv", index=False)

reg = pd.DataFrame(register)
order = ["issue_id", "table", "fields", "issue", "affected_rows", "affected_pct",
         "severity", "evidence", "rule", "resolution"]
reg = reg[order]
reg.to_csv(f"{OUT}/data_quality_register.csv", index=False)

print("=" * 78)
print("REPAIR SUMMARY")
print("=" * 78)
print(f"customer_info : {n_raw:,} -> {len(ci_out):,} rows "
      f"({n_raw - len(ci_out):,} duplicate rows removed)")
print(f"usage_data    : {len(ud_raw):,} -> {len(ud_out):,} rows (none removed, "
      f"{n_pre:,} flagged)")
print(f"churn_labels  : {len(cl):,} rows (unchanged)")
print()
print(reg[["issue_id", "fields", "affected_rows", "affected_pct", "severity",
           "resolution"]].to_string(index=False))

# ------------------------------------------------- before / after impact
print()
print("=" * 78)
print("IMPACT ON HEADLINE METRICS: naive vs audited")
print("=" * 78)


def naive():
    m = ci_raw.merge(cl, on="CustomerID")
    return {
        "rows after joining customer_info to churn_labels": len(m),
        "churn_rate": m.Churn.mean(),
        "mean_MonthlyCharges": m.MonthlyCharges.mean(),
        "mean_Age": m.Age.mean(),
    }


def audited():
    m = ci_out.merge(cl, on="CustomerID")
    return {
        "rows after joining customer_info to churn_labels": len(m),
        "churn_rate": m.Churn.mean(),
        "mean_MonthlyCharges": m.MonthlyCharges_clean.mean(),
        "mean_Age": m.Age_imputed.mean(),
    }


n, a = naive(), audited()
rows = []
for k in n:
    if "churn_rate" in k:
        rows.append([k, f"{n[k]:.4%}", f"{a[k]:.4%}", f"{(a[k]-n[k])*100:+.3f} pp"])
    elif "rows" in k:
        rows.append([k, f"{n[k]:,}", f"{a[k]:,}", f"{a[k]-n[k]:+,}"])
    else:
        rows.append([k, f"{n[k]:.3f}", f"{a[k]:.3f}", f"{a[k]-n[k]:+.3f}"])
imp = pd.DataFrame(rows, columns=["metric", "naive", "audited", "change"])
print(imp.to_string(index=False))
imp.to_csv(f"{OUT}/impact_naive_vs_audited.csv", index=False)

print()
print("Note: the naive join silently inflates every customer who appears twice in")
print("customer_info, so its churn rate and averages are wrong. Deduplicating moves")
print(f"the churn rate from {n['churn_rate']:.4%} to {a['churn_rate']:.4%} and mean charges")
print(f"from {n['mean_MonthlyCharges']:.3f} to {a['mean_MonthlyCharges']:.3f}.")

assert pd.read_csv(f"{CLEAN}/customer_info_clean.csv").CustomerID.is_unique
print("\nverified: cleaned customer_info has unique CustomerID.")
print(f"register rows written: {len(reg)}")
