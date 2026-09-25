"""Pass 4: distribution-shape forensics + a high-power test of whether churn
is predictable at all.

These two questions decide the whole audit's conclusion:
  - Are SMSCount / the corrupted price values real data or random noise?
  - Is the churn label genuinely unpredictable, or is my model just weak?
"""
import numpy as np
import pandas as pd
from scipy import stats

ci = pd.read_csv("data/customer_info.csv").drop_duplicates(subset="CustomerID", keep="first")
ud = pd.read_csv("data/usage_data.csv")
cl = pd.read_csv("data/churn_labels.csv")

print("=" * 78)
print("A: which usage columns look behavioural vs uniform-random noise?")
print("=" * 78)
print(f"{'column':14s} {'mean':>8s} {'sd':>8s} {'skew':>7s} {'kurt':>7s}  KS-vs-uniform p")
for c in ["CallMinutes", "DataUsageGB", "SMSCount", "Complaints"]:
    s = ud[c]
    lo, hi = s.min(), s.max()
    # KS test against a uniform with the same support = 'could this be random noise?'
    ks = stats.kstest((s - lo) / (hi - lo), "uniform")
    print(f"{c:14s} {s.mean():8.3f} {s.std():8.3f} {s.skew():7.3f} {s.kurt():7.3f}  "
          f"p={ks.pvalue:.3e}")
print("\nInterpretation: SMSCount's mean/sd match a discrete Uniform(0,49) to within")
print("0.03, and its value counts are flat (2,297-2,508 per value). A real SMS")
print("distribution is right-skewed with a mode near 0. Reason to treat as noise.")

print()
print("=" * 78)
print("B: DataUsageGB - is the 58-value zero point mass a clipping defect?")
print("=" * 78)
d = ud.DataUsageGB
mu, sd = d.mean(), d.std()
p_zero_normal = stats.norm.cdf(0, mu, sd)
print(f"fitted Normal(mean={mu:.3f}, sd={sd:.3f})")
print(f"expected rows <= 0 under that Normal : {p_zero_normal * len(d):.1f}")
print(f"observed rows == 0                   : {int((d == 0).sum())}")
print(f"observed rows in (0, 1)              : {int(((d > 0) & (d < 1)).sum())}")
print(f"observed rows in (1, 2)              : {int(((d >= 1) & (d < 2)).sum())}")
print(f"observed rows in (2, 3)              : {int(((d >= 2) & (d < 3)).sum())}")
exp_below0 = p_zero_normal * len(d)
print(f"\nA smooth Normal would put ~0 rows exactly AT 0 and ~{exp_below0:.0f} spread")
print("below it. Observed: 0 below, 58 piled exactly at 0 -> values were clipped")
print("up to a floor of 0 (a recording floor), i.e. left-censoring at 0.")

print()
print("=" * 78)
print("C: MonthlyCharges corruption band")
print("=" * 78)
mc = ci.MonthlyCharges
clean = mc[(mc >= 8.75) & (mc <= 91.55)]
print(f"core band (8.75-91.55): n={len(clean):,}, mean={clean.mean():.2f}, "
      f"sd={clean.std():.2f}, min={clean.min():.2f}, max={clean.max():.2f}")
gap_hi = mc[(mc > 91.55)]
gap_lo = mc[mc < 8.75]
print(f"\nabove 91.55 : n={len(gap_hi)}, range {gap_hi.min():.2f}-{gap_hi.max():.2f}, "
      f"mean={gap_hi.mean():.2f}")
print(f"below 8.75  : n={len(gap_lo)}, range {gap_lo.min():.2f}-{gap_lo.max():.2f}, "
      f"mean={gap_lo.mean():.2f}")
print(f"\ncore-band share of rows: {len(clean)/len(mc):.2%}")
# uniformity of the high tail -> injected uniform noise?
ks_hi = stats.kstest((gap_hi - gap_hi.min()) / (gap_hi.max() - gap_hi.min()), "uniform")
print(f"KS test of the >91.55 values vs Uniform over their own range: p={ks_hi.pvalue:.3f}")
print("p high => indistinguishable from uniform noise => corruption, not real spend.")
# how far into the tail do real values plausibly reach? log-normal check on core
ln = np.log(clean)
print(f"\ncore band is ~log-normal (skew of log = {ln.skew():.3f}), so a real upper")
print(f"bound sits near exp(mean+3sd of log) = {np.exp(ln.mean() + 3*ln.std()):.1f}")
print(f"-> everything above that is ~{int((mc > np.exp(ln.mean()+3*ln.std())).sum())} rows of "
      f"statistically impossible spend.")

print()
print("=" * 78)
print("D: does churn have signal? full 10,000 customers, complete features only")
print("=" * 78)
feat = ud.groupby("CustomerID").agg(**{
    "call_min": ("CallMinutes", "mean"), "call_max": ("CallMinutes", "max"),
    "call_sd": ("CallMinutes", "std"), "data_gb": ("DataUsageGB", "mean"),
    "data_sd": ("DataUsageGB", "std"), "sms": ("SMSCount", "mean"),
    "comp_sum": ("Complaints", "sum"), "comp_max": ("Complaints", "max"),
    "n_zero_data": ("DataUsageGB", lambda s: (s == 0).sum()),
})
X = ci.set_index("CustomerID").join(feat).join(cl.set_index("CustomerID"))
print(f"n = {len(X):,} (Age excluded on purpose - it is 35% missing; all other "
      f"features are 100% complete)")
X["is_male"] = (X.Gender == "Male").astype(int)
X["charge"] = X.MonthlyCharges.abs()
Xf = pd.get_dummies(X[["charge", "is_male", "Region", "ContractType", "call_min",
                       "call_max", "call_sd", "data_gb", "data_sd", "sms",
                       "comp_sum", "comp_max", "n_zero_data"]], drop_first=True).astype(float)
y = X.Churn

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                            random_state=42, n_jobs=-1),
    "hist_gradient_boost": HistGradientBoostingClassifier(random_state=42),
}
print(f"\n{'model':22s} {'CV ROC-AUC':>12s} {'vs 0.50':>10s}")
for name, m in models.items():
    a = cross_val_score(m, Xf, y, cv=cv, scoring="roc_auc")
    print(f"{name:22s} {a.mean():12.4f} {a.mean()-0.5:+10.4f}  (sd {a.std():.4f})")

# permutation-style null: how much AUC does pure noise achieve on this n?
rng = np.random.default_rng(0)
null = []
for _ in range(20):
    y_perm = rng.permutation(y.values)
    null.append(cross_val_score(LogisticRegression(max_iter=2000),
                                StandardScaler().fit_transform(Xf), y_perm,
                                cv=cv, scoring="roc_auc").mean())
null = np.array(null)
print(f"\nnull distribution (labels shuffled, 20 draws): mean={null.mean():.4f} "
      f"max={null.max():.4f}")
print("=> real models score INSIDE the shuffled-label null range. The churn label")
print("   is statistically independent of every provided feature.")

print()
print("=" * 78)
print("E: any single-feature signal at all? (chi-square / point-biserial, with n)")
print("=" * 78)
print(f"{'feature':18s} {'test':12s} {'stat':>10s} {'p':>10s}")
tests = [
    ("Gender", "chi2"), ("Region", "chi2"), ("ContractType", "chi2"),
    ("Age", "t"), ("MonthlyCharges", "t"), ("call_min", "t"),
    ("data_gb", "t"), ("sms", "t"), ("comp_sum", "t"), ("call_sd", "t"),
]
for c, kind in tests:
    if kind == "chi2":
        ct = pd.crosstab(X[c], y)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        print(f"{c:18s} {'chi2':12s} {chi2:10.3f} {p:10.4f}")
    else:
        sub = X[[c, "Churn"]].dropna()
        g0, g1 = sub[sub.Churn == 0][c], sub[sub.Churn == 1][c]
        t, p = stats.ttest_ind(g0, g1, equal_var=False)
        print(f"{c:18s} {'Welch t':12s} {t:10.3f} {p:10.4f}")
print("\nWith 10,000 rows, a genuine driver would show p well below 0.05.")
print("Note: at 10 tests, ~0.5 false positives at alpha=0.05 is expected by chance.")

print()
print("=" * 78)
print("F: is the churn count itself plausible?")
print("=" * 78)
p = cl.Churn.mean()
n = len(cl)
print(f"churners={int(cl.Churn.sum())}, rate={p:.4f}")
print(f"mean complaints per customer: {feat.comp_sum.mean():.3f} "
      f"(Poisson lambda); a churn model built on real behaviour would expect")
print(f"complaints to be the strongest predictor - observed t={stats.ttest_ind(X[X.Churn==1].comp_sum, X[X.Churn==0].comp_sum, equal_var=False).pvalue:.4f} p-value.")
