"""GAP CHECK: did we miss the trend signal?

The brief states churn patterns are behaviourally embedded as a usage DECLINE
before churn. Our original feature set used means / SDs / maxes over the 12
months. A standard deviation measures volatility, NOT direction - a steady rise
and a steady decline have identical SDs. So a decline-before-churn signal would
have been structurally invisible to the original model.

This test engineers trend features and re-runs the signal test.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ud = pd.read_csv("data/usage_data.csv")
ci = pd.read_csv("data/customer_info.csv").drop_duplicates(subset="CustomerID", keep="first")
cl = pd.read_csv("data/churn_labels.csv")

ud["t"] = pd.to_datetime(ud.Month).dt.month  # 1..12, evenly spaced

print("=" * 78)
print("A: is there a usage DECLINE for churners? (raw look, no model)")
print("=" * 78)
y = cl.set_index("CustomerID").Churn

for col in ["CallMinutes", "DataUsageGB", "Complaints", "SMSCount"]:
    print(f"\n--- {col}: mean per calendar month, split by churn ---")
    p = ud.pivot_table(index="t", columns="CustomerID", values=col)
    churners = [c for c in p.columns if y.get(c, 0) == 1]
    stayers = [c for c in p.columns if y.get(c, 0) == 0]
    tab = pd.DataFrame({
        "churn=0": p[stayers].mean(axis=1),
        "churn=1": p[churners].mean(axis=1),
    })
    tab["diff"] = tab["churn=1"] - tab["churn=0"]
    tab["diff_%"] = tab["diff"] / tab["churn=0"] * 100
    print(tab.round(2).to_string())
    d1, d12 = tab.loc[1, "diff"], tab.loc[12, "diff"]
    print(f"    gap at month 1: {d1:+.2f}   gap at month 12: {d12:+.2f}   "
          f"change: {d12 - d1:+.2f}")

print()
print("=" * 78)
print("B: engineer TREND features and re-run the signal test")
print("=" * 78)


def slope(v):
    """OLS slope of a 12-point series against time - the 'decline' signal."""
    t = np.arange(len(v))
    return np.polyfit(t, v, 1)[0]


def half_ratio(v):
    """mean(last 3 months) / mean(first 3 months) - the 'decline' as a ratio."""
    a, b = v[:3].mean(), v[-3:].mean()
    return b / a if a > 0 else np.nan


def feats(g):
    g = g.sort_values("t")
    out = {}
    for col, tag in [("CallMinutes", "call"), ("DataUsageGB", "data"),
                     ("SMSCount", "sms"), ("Complaints", "comp")]:
        v = g[col].to_numpy(float)
        out[f"{tag}_slope"] = slope(v)
        out[f"{tag}_mean"] = v.mean()
        out[f"{tag}_sd"] = v.std()
        out[f"{tag}_first3"] = v[:3].mean()
        out[f"{tag}_last3"] = v[-3:].mean()
        out[f"{tag}_halfratio"] = half_ratio(v)
        out[f"{tag}_last_minus_first"] = v[-1] - v[0]
        out[f"{tag}_last1"] = v[-1]
        out[f"{tag}_max"] = v.max()
    return pd.Series(out)


F = ud.groupby("CustomerID").apply(feats, include_groups=False)
print(f"engineered {F.shape[1]} features for {len(F):,} customers")

X = ci.set_index("CustomerID").join(F).join(y)
X["is_male"] = (X.Gender == "Male").astype(int)
X["charge"] = X.MonthlyCharges.abs()

# ---- univariate: does each trend feature differ by churn?
print("\nunivariate Welch t-test on TREND features (churn vs not):")
rows = []
for c in [c for c in F.columns if "slope" in c or "halfratio" in c or "last_minus_first" in c]:
    s = X[[c, "Churn"]].dropna()
    t, p = stats.ttest_ind(s[s.Churn == 1][c], s[s.Churn == 0][c], equal_var=False)
    r = s[c].corr(s.Churn)
    rows.append((c, t, p, r))
uni = pd.DataFrame(rows, columns=["feature", "t", "p", "r"]).sort_values("p")
print(uni.to_string(index=False))

# ---- multivariable
feat_cols = ["charge", "is_male", "Age"] + [c for c in F.columns]
Xd = pd.get_dummies(X[feat_cols + ["Region", "ContractType"]],
                    columns=["Region", "ContractType"], drop_first=True).astype(float)
Xd = Xd.fillna(Xd.median())
yv = X.Churn

cv = StratifiedKFold(5, shuffle=True, random_state=42)
print("\n5-fold CV ROC-AUC with TREND features included:")
for name, m in {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                            random_state=42, n_jobs=-1),
}.items():
    a = cross_val_score(m, Xd, yv, cv=cv, scoring="roc_auc")
    print(f"   {name:15s} AUC = {a.mean():.4f} (+/- {a.std():.4f})")

print("\nnull band for reference (labels shuffled, 10 draws):")
rng = np.random.default_rng(0)
null = [cross_val_score(LogisticRegression(max_iter=2000),
                        StandardScaler().fit_transform(Xd), rng.permutation(yv.values),
                        cv=cv, scoring="roc_auc").mean() for _ in range(10)]
print(f"   {np.min(null):.4f} - {np.max(null):.4f}")

F.to_csv("outputs/trend_features.csv")
uni.to_csv("outputs/trend_univariate.csv", index=False)
print("\nwrote outputs/trend_features.csv + outputs/trend_univariate.csv")
