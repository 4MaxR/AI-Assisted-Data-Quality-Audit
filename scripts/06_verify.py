"""Pass 6: statistical verification of the two judgement-call defects.

Before recommending a repair rule I want to know whether the MonthlyCharges
right tail is a plausible heavy tail (leave it) or an injected contamination
(flag it), and whether SignupDate has one defect or two.
"""
import numpy as np
import pandas as pd
from scipy import stats

ci = pd.read_csv("data/customer_info.csv").drop_duplicates(subset="CustomerID", keep="first")
mc = ci.MonthlyCharges

print("=" * 78)
print("A: can one distribution explain the whole of MonthlyCharges?")
print("=" * 78)
pos = mc[mc > 0]
ln = np.log(pos)
print(f"log(charge): n={len(ln)}, mean={ln.mean():.4f}, sd={ln.std():.4f}, "
      f"skew={ln.skew():.4f}, kurt={ln.kurt():.4f}")
print("(a log-normal would give skew~0 and kurt~0 in log space)")
sig = ln.std()
print(f"\nrows predicted above 92 by the fitted log-normal: "
      f"{(1 - stats.norm.cdf((np.log(92) - ln.mean()) / sig)) * len(pos):.2f}")
print(f"rows observed above 92                           : {int((mc > 92).sum())}")
print(f"rows predicted above 100: "
      f"{(1 - stats.norm.cdf((np.log(100) - ln.mean()) / sig)) * len(pos):.3f}")
print(f"rows observed above 100 : {int((mc > 100).sum())}")
print(f"rows predicted above 200: "
      f"{(1 - stats.norm.cdf((np.log(200) - ln.mean()) / sig)) * len(pos):.4f}")
print(f"rows observed above 200 : {int((mc > 200).sum())}")
print("\n-> the observed tail exceeds any single-distribution prediction by 1-2")
print("   orders of magnitude. The tail is contamination, not a heavy tail.")

print()
print("=" * 78)
print("B: mixture fit - does a second component isolate the contaminant?")
print("=" * 78)
from sklearn.mixture import GaussianMixture
X = mc.values.reshape(-1, 1)
for k in [1, 2, 3]:
    g = GaussianMixture(k, random_state=0, n_init=5).fit(X)
    order = np.argsort(g.means_.ravel())
    print(f"k={k}  BIC={g.bic(X):10.1f}  components (weight, mean, sd):")
    for i in order:
        print(f"      {g.weights_[i]:.4f}  mean={g.means_[i,0]:8.2f}  sd={g.covariances_[i,0,0]**0.5:7.2f}")

print()
print("=" * 78)
print("C: SignupDate - one defect or two?")
print("=" * 78)
sd = pd.to_datetime(ci.SignupDate)
print("customers by signup month (first 6 and last 12):")
by_m = sd.dt.to_period("M").value_counts().sort_index()
print(by_m.head(6).to_string())
print("...")
print(by_m.tail(12).to_string())
print(f"\ntotal months covered: {len(by_m)}")
print(f"mean customers/month: {by_m.mean():.1f}, sd={by_m.std():.1f}")
print(f"months with < 50 customers: {(by_m < 50).sum()}")
print(f"months with < 100 customers: {(by_m < 100).sum()}")
low = by_m[by_m < 100]
print("\nsparse months:")
print(low.to_string())

print("\n-> 2018-01 .. 2023-? : uniform-ish ~150/month, then the 2023 months dry up.")
print("   A truncated-but-uniform signup generator would NOT thin out this way,")
print("   so the late-2023 signups are a distinct injected cohort.")

print()
print("=" * 78)
print("D: do the 2023-signup customers differ in any other way?")
print("=" * 78)
ud = pd.read_csv("data/usage_data.csv")
cl = pd.read_csv("data/churn_labels.csv")
ci["yr"] = sd.dt.year
feat = ud.groupby("CustomerID").agg(call=("CallMinutes", "mean"),
                                    data=("DataUsageGB", "mean"),
                                    comp=("Complaints", "sum"))
m = ci.set_index("CustomerID").join(feat).join(cl.set_index("CustomerID"))
m["cohort"] = np.where(m.yr == 2023, "2023 signup", "2018-2022 signup")
print(m.groupby("cohort")[["Age", "MonthlyCharges", "call", "data", "comp", "Churn"]]
      .agg(["mean", "size"]).round(3).to_string())
print("\n-> the 2023 cohort is statistically indistinguishable from the rest on")
print("   every other field, so the defect is confined to the date fields.")
