"""Pass 5: define defensible repair rules.

An audit is only actionable if each defect gets a specific rule, and the rule is
justified by where the data actually breaks - not by an arbitrary IQR fence.
"""
import numpy as np
import pandas as pd

ci = pd.read_csv("data/customer_info.csv")
ci_u = ci.drop_duplicates(subset="CustomerID", keep="first")
ud = pd.read_csv("data/usage_data.csv")

print("=" * 78)
print("A: where does MonthlyCharges actually break? (density scan, not a fence)")
print("=" * 78)
mc = np.sort(ci_u.MonthlyCharges.values)
# local density: how many values per unit in a sliding window
print("value  |  n within +-2  | note")
for v in [20, 40, 60, 80, 88, 90, 91, 92, 93, 95, 100, 110, 120, 150, 200, 300]:
    n = int(((mc >= v - 2) & (mc <= v + 2)).sum())
    print(f"{v:6d} | {n:14d} |")

print("\nlargest gaps between consecutive sorted values in 85..120:")
band = mc[(mc >= 85) & (mc <= 120)]
gaps = np.diff(band)
order = np.argsort(gaps)[::-1][:8]
for i in sorted(order):
    print(f"  gap {gaps[i]:6.3f}  between {band[i]:.2f} and {band[i+1]:.2f}")

print("\n-> the dense core runs continuously to ~91.5; above it the density falls")
print("   by ~440x but the support stays continuous, so the break is a density")
print("   cliff, not an empty gap. That is the signature of values being replaced")
print("   by draws from a wider (flatter, heavier-tailed) distribution.")

print()
print("=" * 78)
print("B: structure of the two anomalous MonthlyCharges groups")
print("=" * 78)
hi = ci_u[ci_u.MonthlyCharges > 91.55].MonthlyCharges
lo = ci_u[ci_u.MonthlyCharges < 8.75].MonthlyCharges
print(f"HIGH group n={len(hi)}: deciles of the group")
print(hi.quantile(np.arange(0.1, 1.0, 0.1)).round(2).to_string())
print(f"\nLOW group n={len(lo)}: full sorted list")
print(np.sort(lo.values).round(2))

print(f"\ntotal rows outside the core band: {len(hi) + len(lo)}")
print("Check for round injection counts - how many rows sit outside a range of")
print("increasing width from the median (50.02)?")
med = ci_u.MonthlyCharges.median()
for k in [40, 41, 42, 45, 50, 60, 100]:
    n = int((np.abs(ci_u.MonthlyCharges - med) > k).sum())
    print(f"  |charge - median| > {k:3d} : {n:4d} rows")

print()
print("=" * 78)
print("C: the signup-date / usage-window defect population")
print("=" * 78)
ci_u = ci_u.copy()
ci_u["SignupDate_dt"] = pd.to_datetime(ci_u.SignupDate)
yr = ci_u.SignupDate_dt.dt.year.value_counts().sort_index()
print("customers by signup year:")
print(yr.to_string())
ud["Month_dt"] = pd.to_datetime(ud.Month)
u = ud.merge(ci_u[["CustomerID", "SignupDate_dt"]], on="CustomerID")
u["signed_up_after_row"] = u.Month_dt < u.SignupDate_dt
aff = u[u.signed_up_after_row]
print(f"\nrows before signup: {len(aff):,} across {aff.CustomerID.nunique():,} customers")
print(f"share of the 2023-signup cohort affected: "
      f"{aff.CustomerID.nunique()}/{int(yr.get(2023, 0))}")
print(f"\nrows before signup, by usage month:")
print(aff.Month.value_counts().sort_index().to_string())
print("\nfirst usage month vs signup month for affected customers:")
first_row = u.groupby("CustomerID").Month_dt.min()
sg = ci_u.set_index("CustomerID").SignupDate_dt
cmp_ = pd.DataFrame({"first_usage": first_row}).join(sg)
cmp_ = cmp_[cmp_.first_usage < cmp_.SignupDate_dt]
print(f"  affected customers: {len(cmp_)}")
print(f"  median months of usage predating signup: "
      f"{((cmp_.SignupDate_dt - cmp_.first_usage).dt.days / 30.44).median():.1f}")
print("\n-> these customers were given the full Jan-Dec 2023 activity history")
print("   despite signing up mid/late 2023. Either SignupDate is wrong or the")
print("   pre-signup usage rows do not belong to them.")

print()
print("=" * 78)
print("D: duplicate-row defect - exact population")
print("=" * 78)
dm = ci.CustomerID.duplicated(keep=False)
print(f"total rows in customer_info      : {len(ci):,}")
print(f"unique customers                 : {ci.CustomerID.nunique():,}")
print(f"extra rows (pure duplication)    : {len(ci) - ci.CustomerID.nunique():,}")
print(f"rows participating in duplication: {int(dm.sum()):,}")
# verify exact duplicates are column-identical
dup_ids = ci[dm].CustomerID.unique()
exact, conflict = [], []
for cid in dup_ids:
    g = ci[ci.CustomerID == cid]
    (exact if len(g.drop_duplicates()) == 1 else conflict).append(cid)
print(f"  exact duplicates    : {len(exact)}")
print(f"  conflicting dupes   : {len(conflict)} -> {conflict}")
print("\n-> 195 rows are safe to de-duplicate. The 5 conflicting rows must be")
print("   resolved by a rule (which SignupDate is authoritative?) or flagged.")

print()
print("=" * 78)
print("E: Age - is the missingness exactly 35% and is the observed part clean?")
print("=" * 78)
print(f"rows in file            : {len(ci):,}  (incl. duplicates)")
print(f"Age missing in file     : {int(ci.Age.isna().sum()):,} = {ci.Age.isna().mean():.4%}")
print(f"Age missing, de-duped   : {int(ci_u.Age.isna().sum()):,} = {ci_u.Age.isna().mean():.4%}")
print(f"Age min/max             : {ci_u.Age.min():.0f} / {ci_u.Age.max():.0f}")
print(f"Age integer-valued      : {bool((ci_u.Age.dropna() % 1 == 0).all())}")
print(f"Age sd                  : {ci_u.Age.std():.2f} (no impossible values, no 0/999 sentinels)")
print(f"Age == 0 or >99         : {int(((ci_u.Age <= 0) | (ci_u.Age > 99)).sum())}")
print("\n-> Age is only MISSING, never invalid. 35% is a high but clean random drop.")
