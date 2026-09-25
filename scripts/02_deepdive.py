"""Pass 2: deep dive into specific anomalies found in pass 1.

Each check states a hypothesis about WHAT KIND of defect it is, because the
remediation differs: exact-duplicate rows (dedupe) vs conflicting-duplicate rows
(needs a rule) vs missing-not-at-random (needs investigation) vs out-of-range
(clamp/flag) vs impossible-timeline (referential/logic violation).
"""
import pandas as pd
import numpy as np

ci = pd.read_csv("data/customer_info.csv")
ud = pd.read_csv("data/usage_data.csv")
cl = pd.read_csv("data/churn_labels.csv")

print("=" * 78)
print("CHECK 1: duplicate CustomerIDs in customer_info  (200 rows, 195 full dupes)")
print("=" * 78)
dupmask = ci.CustomerID.duplicated(keep=False)
dups = ci[dupmask].sort_values("CustomerID")
print(f"rows involved in duplicate IDs : {len(dups)}")
print(f"distinct duplicated CustomerIDs: {dups.CustomerID.nunique()}")

# split: exact duplicates (every col identical) vs conflicting duplicates
grp = dups.groupby("CustomerID")
exact_ids, conflict_ids = [], []
for cid, g in grp:
    # compare ignoring NaN placement differences
    if len(g.drop_duplicates()) == 1:
        exact_ids.append(cid)
    else:
        conflict_ids.append(cid)
print(f"  -> exactly-identical duplicate groups : {len(exact_ids)}  (ids {exact_ids[:5]}...)")
print(f"  -> CONFLICTING duplicate groups       : {len(conflict_ids)}  (ids {conflict_ids})")

if conflict_ids:
    print("\nConflicting duplicate rows (all columns shown):")
    print(dups[dups.CustomerID.isin(conflict_ids)].to_string(index=False))
    print("\nWhich columns actually disagree, per conflicting ID:")
    for cid, g in dups[dups.CustomerID.isin(conflict_ids)].groupby("CustomerID"):
        g2 = g.drop_duplicates()
        diffcols = [c for c in ci.columns if g2[c].nunique(dropna=False) > 1]
        print(f"  {cid}: differing columns = {diffcols}")

print()
print("=" * 78)
print("CHECK 2: Age missingness pattern (35.1% missing)")
print("=" * 78)
ci["_age_missing"] = ci.Age.isna()
# dedupe first so the duplicated rows don't skew rates
ci_d = ci.drop_duplicates(subset="CustomerID", keep="first")
print(f"using {len(ci_d):,} unique customers")
print(f"overall Age missing rate: {ci_d.Age.isna().mean():.4f}")
print()
for col in ["Gender", "Region", "ContractType"]:
    t = ci_d.groupby(col, dropna=False)["_age_missing"].agg(["mean", "size"])
    t.columns = ["missing_rate", "n"]
    print(f"by {col}:")
    print(t.round(4).to_string())
    print()
# is missingness related to churn?
m = ci_d.merge(cl, on="CustomerID")
print("by churn label:")
print(m.groupby("Churn")["_age_missing"].agg(["mean", "size"]).round(4).to_string())
print()
# are non-missing ages integers?
ages = ci_d.Age.dropna()
print(f"non-integer Age values: {int((ages != ages.round()).sum())}")
print(f"Age unique values: {sorted(ages.unique())[:8]} ... {sorted(ages.unique())[-4:]}")

print()
print("=" * 78)
print("CHECK 3: MonthlyCharges negatives + outliers")
print("=" * 78)
ci_d["MonthlyCharges"] = ci_d["MonthlyCharges"]
neg = ci_d[ci_d.MonthlyCharges < 0]
print(f"negative MonthlyCharges: {len(neg)} rows")
print(neg[["CustomerID", "MonthlyCharges", "ContractType", "Region"]].to_string(index=False))
q1, q3 = ci_d.MonthlyCharges.quantile([0.25, 0.75])
iqr = q3 - q1
hi = q3 + 1.5 * iqr
print(f"\nQ1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} upper fence={hi:.2f}")
top = ci_d.nlargest(10, "MonthlyCharges")[["CustomerID", "MonthlyCharges"]]
print("top 10 charges:")
print(top.to_string(index=False))
print(f"charges above fence: {(ci_d.MonthlyCharges > hi).sum()}")
print(f"charges below lower fence: {(ci_d.MonthlyCharges < q1 - 1.5 * iqr).sum()}")

print()
print("=" * 78)
print("CHECK 4: SignupDate vs usage window (impossible timeline?)")
print("=" * 78)
ci_d["SignupDate_dt"] = pd.to_datetime(ci_d.SignupDate, errors="coerce")
print(f"unparseable SignupDate: {int(ci_d.SignupDate_dt.isna().sum())}")
print(f"SignupDate range: {ci_d.SignupDate_dt.min().date()} -> {ci_d.SignupDate_dt.max().date()}")
ud["Month_dt"] = pd.to_datetime(ud.Month)
u = ud.merge(ci_d[["CustomerID", "SignupDate_dt", "SignupDate"]], on="CustomerID")
early = u[u.Month_dt < u.SignupDate_dt]
print(f"\nusage rows dated BEFORE the customer's SignupDate: {len(early):,} "
      f"({len(early)/len(u)*100:.2f}% of usage rows)")
print(f"customers affected: {early.CustomerID.nunique():,} of {u.CustomerID.nunique():,}")
if len(early):
    lag = (early.SignupDate_dt - early.Month_dt).dt.days
    print(f"median days before signup: {lag.median():.0f}, max: {lag.max()}")
    print("\nexample rows:")
    print(early[["CustomerID", "Month", "SignupDate", "CallMinutes"]].head(8).to_string(index=False))
print(f"\nSignupDate after the usage window ends (2023-12-31): "
      f"{(ci_d.SignupDate_dt > pd.Timestamp('2023-12-31')).sum()} customers")

print()
print("=" * 78)
print("CHECK 5: usage_data outliers / suspicious distributions")
print("=" * 78)
for c in ["CallMinutes", "DataUsageGB", "SMSCount", "Complaints"]:
    s = ud[c]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std()
    print(f"{c:12s} mean={s.mean():8.2f} sd={s.std():7.2f} min={s.min():8.2f} max={s.max():8.2f} "
          f"| IQR fence [{lo:8.2f},{hi:8.2f}] n_out={int(((s<lo)|(s>hi)).sum()):6,} "
          f"| |z|>4: {int((z.abs()>4).sum()):5,}")
print()
print("SMSCount distribution (is 49 a hard cap?):")
print(ud.SMSCount.value_counts().sort_index().tail(6).to_string())
print("\nComplaints distribution:")
print(ud.Complaints.value_counts().sort_index().to_string())
print("\nDataUsageGB == 0 rows (possible disguised missing):")
print(f"  count: {(ud.DataUsageGB == 0).sum()}, customers: {ud[ud.DataUsageGB==0].CustomerID.nunique()}")
print(f"  do these rows have other activity? mean CallMinutes = "
      f"{ud[ud.DataUsageGB==0].CallMinutes.mean():.2f} vs overall {ud.CallMinutes.mean():.2f}")

print()
print("=" * 78)
print("CHECK 6: Month column hygiene / completeness")
print("=" * 78)
months = pd.to_datetime(ud.Month)
print(f"all Months are end-of-month? {(months == months + pd.offsets.MonthEnd(0)).all()}")
print(f"n unique months: {months.nunique()}, expected 12 contiguous: "
      f"{months.nunique() == 12}")
full = pd.date_range(months.min(), months.max(), freq="ME")
print(f"gap check - missing months: {sorted(set(full) - set(months))}")
print(f"customer x month grid complete? rows={len(ud):,} vs 10,000*12={10000*12:,}")
print(f"duplicate (CustomerID, Month) pairs: {ud.duplicated(['CustomerID','Month']).sum()}")

print()
print("=" * 78)
print("CHECK 7: categorical hygiene (whitespace / casing variants)")
print("=" * 78)
for c in ["Gender", "Region", "ContractType"]:
    raw = ci_d[c].astype(str)
    print(f"{c}: raw unique = {sorted(raw.unique())}")
    print(f"    differs after strip+lower? {sorted(raw.unique()) != sorted(raw.str.strip().str.lower().unique())}"
          f"  | any leading/trailing space: {raw.ne(raw.str.strip()).any()}")

print()
print("=" * 78)
print("CHECK 8: is churn label consistent with a usable target?")
print("=" * 78)
m = ci_d.merge(ud.groupby("CustomerID").agg(
    avg_complaints=("Complaints", "mean"),
    avg_data=("DataUsageGB", "mean"),
    tenure_months=("Month", "nunique"),
), on="CustomerID").merge(cl, on="CustomerID")
print(f"customers: {len(m):,}  churn rate: {m.Churn.mean():.4f}")
print("\nchurn rate by ContractType (a plausible real driver - sanity check):")
print(m.groupby("ContractType").Churn.agg(["mean", "size"]).round(4).to_string())
print("\nchurn rate by complaints level:")
m["complaint_band"] = pd.cut(m.avg_complaints, [-0.01, 0.5, 1.5, 3, 100],
                             labels=["<=0.5", "0.5-1.5", "1.5-3", ">3"])
print(m.groupby("complaint_band", observed=True).Churn.agg(["mean", "size"]).round(4).to_string())
print("\nNote: every customer has tenure_months == 12 (no tenure variation).")
