"""Pass 3: characterise the anomalies and test whether the data supports
modelling at all.

Key question for a data quality audit: are the defects fixable, and once fixed,
does the dataset carry the signal it claims to (churn)? A clean-but-signal-free
dataset is a different finding than a dirty one.
"""
import numpy as np
import pandas as pd

ci = pd.read_csv("data/customer_info.csv").drop_duplicates(subset="CustomerID", keep="first")
ud = pd.read_csv("data/usage_data.csv")
cl = pd.read_csv("data/churn_labels.csv")

print("=" * 78)
print("A: MonthlyCharges - are the high values a decimal-shift defect?")
print("=" * 78)
mc = ci.MonthlyCharges
print(f"percentiles: 1%={mc.quantile(.01):.2f} 25%={mc.quantile(.25):.2f} "
      f"50%={mc.quantile(.50):.2f} 75%={mc.quantile(.75):.2f} 99%={mc.quantile(.99):.2f}")
hist = pd.cut(mc, [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400]).value_counts().sort_index()
print("\nhistogram of MonthlyCharges:")
for k, v in hist.items():
    print(f"  {str(k):>16s} : {v:5,}  {'#' * int(v / 60)}")

high = ci[mc > 91.55].copy()
print(f"\n{len(high)} rows above the IQR fence (91.55)")
print("are they a contiguous tail or a separate cluster? value range: "
      f"{high.MonthlyCharges.min():.2f} -> {high.MonthlyCharges.max():.2f}")
div10 = high.MonthlyCharges / 10
print(f"if divided by 10: range {div10.min():.2f} -> {div10.max():.2f}; "
      f"share landing in the normal 20-80 band: {((div10 >= 20) & (div10 <= 80)).mean():.1%}")
print("-> a clean /10 landing would indicate a decimal-shift data-entry error;")
print("   a long smooth tail instead indicates genuine high-value customers.")
print("\nsample of high values vs their /10:")
print(pd.DataFrame({"as_recorded": high.MonthlyCharges.head(8).values,
                    "divided_by_10": (high.MonthlyCharges.head(8) / 10).round(2).values}
                   ).to_string(index=False))

low = ci[mc < 8.75]
print(f"\n{len(low)} rows below the low fence / negative:")
print(low[["CustomerID", "MonthlyCharges", "ContractType"]].to_string(index=False))

print()
print("=" * 78)
print("B: SMSCount - is it uniformly random (synthetic) or behavioural?")
print("=" * 78)
vc = ud.SMSCount.value_counts().sort_index()
print(f"min={ud.SMSCount.min()} max={ud.SMSCount.max()} n_unique={ud.SMSCount.nunique()}")
print(f"mean={ud.SMSCount.mean():.3f} (uniform 0-49 would give 24.5)")
print(f"std ={ud.SMSCount.std():.3f} (uniform 0-49 would give 14.43)")
print(f"counts per value: min={vc.min()} max={vc.max()} -> spread {vc.min()}-{vc.max()} "
      f"(uniform expectation 2400)")
print(f"hard cap at 49? values >= 49: {int((ud.SMSCount >= 49).sum())}, "
      f"values == 50: {int((ud.SMSCount == 50).sum())}")
print("\nfirst 12 value counts (should be flat if uniform):")
print(vc.head(12).to_string())

print()
print("=" * 78)
print("C: DataUsageGB - the 2 extreme outliers and the 58 zeros")
print("=" * 78)
z = (ud.DataUsageGB - ud.DataUsageGB.mean()) / ud.DataUsageGB.std()
print("extreme rows (|z|>4):")
print(ud[z.abs() > 4].to_string(index=False))
print("\nDataUsageGB==0 rows: do they otherwise look like dormant customers?")
zero = ud[ud.DataUsageGB == 0]
print(zero[["CallMinutes", "SMSCount", "Complaints"]].describe().round(2).to_string())
print("\nfor contrast, all rows:")
print(ud[["CallMinutes", "SMSCount", "Complaints"]].describe().round(2).to_string())
print("\n-> similar/higher call activity means the 0 is a recording gap, not dormancy.")

print()
print("=" * 78)
print("D: MISSINGNESS MECHANISM for Age (3500/10000 = exactly 35%)")
print("=" * 78)
print("observed missing rate by segment (from pass 2): all within 0.341-0.359.")
n = len(ci)
se = np.sqrt(0.35 * 0.65 / n)
print(f"expected standard error at n={n:,}: {se:.4f} -> +/-3SE = {3*se:.4f}")
print("All segment rates fall inside MCAR sampling noise => consistent with")
print("Age missing completely at random (a clean 35% random drop), NOT")
print("informative missingness driven by region/gender/contract/churn.")

print()
print("=" * 78)
print("E: does the data actually carry churn signal?")
print("=" * 78)
feat = ud.groupby("CustomerID").agg(
    call_min=("CallMinutes", "mean"),
    data_gb=("DataUsageGB", "mean"),
    sms=("SMSCount", "mean"),
    complaints=("Complaints", "mean"),
    call_sd=("CallMinutes", "std"),
    data_sd=("DataUsageGB", "std"),
    complaints_sum=("Complaints", "sum"),
)
X = ci.set_index("CustomerID").join(feat).join(cl.set_index("CustomerID"))
X = X.dropna(subset=["Age"]).copy()
X["is_male"] = (X.Gender == "Male").astype(int)
y = X.Churn

num_cols = ["Age", "MonthlyCharges", "call_min", "data_gb", "sms", "complaints",
            "call_sd", "data_sd", "complaints_sum"]
print("point-biserial correlation of each feature with Churn:")
for c in num_cols:
    r = X[c].corr(y)
    print(f"  {c:16s} r={r:+.4f}")

# quick honest signal test: cross-validated AUC of a shallow model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

Xf = pd.get_dummies(X[["Age", "MonthlyCharges", "Gender", "Region", "ContractType",
                       "call_min", "data_gb", "sms", "complaints", "call_sd",
                       "data_sd", "complaints_sum"]], drop_first=True).astype(float)
Xf = Xf.fillna(Xf.median())
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, mdl in [("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                  ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                           random_state=0, n_jobs=-1))]:
    auc = cross_val_score(mdl, Xf, y, cv=cv, scoring="roc_auc")
    print(f"  {name:14s} 5-fold CV ROC-AUC = {auc.mean():.4f} (+/- {auc.std():.4f})")
print("\nReference: random guessing = 0.5000. AUC near 0.50 means the provided")
print("features do not separate churners from non-churners.")

print()
print("=" * 78)
print("F: alternative explanation - is churn ~ independent Bernoulli?")
print("=" * 78)
p = y.mean()
print(f"observed churn rate: {p:.4f} ({y.sum()}/{len(y)})")
# compare against a binomial with the same p: is the count plausible & are labels shuffled?
ci_grp = X.groupby(pd.qcut(X.complaints_sum, 5, duplicates="drop")).Churn.agg(["mean", "size"])
print("\nchurn rate by complaints quintile (a real churn driver would trend):")
print(ci_grp.round(4).to_string())
ci_grp2 = X.groupby(pd.qcut(X.MonthlyCharges, 5)).Churn.agg(["mean", "size"])
print("\nchurn rate by MonthlyCharges quintile:")
print(ci_grp2.round(4).to_string())
ci_grp3 = X.groupby("ContractType").Churn.agg(["mean", "size"])
print("\nchurn rate by ContractType (Yearly contracts churning MOST is backward):")
print(ci_grp3.round(4).to_string())
ci_grp4 = X.groupby("Region").Churn.agg(["mean", "size"])
print("\nchurn rate by Region:")
print(ci_grp4.round(4).to_string())
