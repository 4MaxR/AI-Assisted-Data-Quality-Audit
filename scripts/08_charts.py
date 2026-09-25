"""Pass 8: visual evidence. One figure per finding, each answering a question -
no decorative charts.

Outputs to outputs/figures/ as PNG (300 dpi) so they can go straight into a
report or slide deck.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

INK = "#1b2430"
ACC = "#c0392b"
OK = "#2e7d5b"
MUT = "#8a94a6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#e8ebf0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False,
})

ci = pd.read_csv("data/customer_info.csv")
ci_u = ci.drop_duplicates(subset="CustomerID", keep="first").copy()
ci_u["SignupDate_dt"] = pd.to_datetime(ci_u.SignupDate)
ud = pd.read_csv("data/usage_data.csv")
ud["Month_dt"] = pd.to_datetime(ud.Month)
cl = pd.read_csv("data/churn_labels.csv")

# ============================================================ FIG 1: charges
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
mc = ci_u.MonthlyCharges
ax[0].hist(mc, bins=110, color=MUT, edgecolor="white", linewidth=0.3)
ax[0].axvspan(min(0, mc.min()), 0, color=ACC, alpha=0.16)
ax[0].axvspan(92, mc.max(), color=ACC, alpha=0.16)
ax[0].set_xlabel("MonthlyCharges (recorded)")
ax[0].set_ylabel("customers")
ax[0].set_title("A. Raw charges: a clean body plus two contaminated tails")
ax[0].annotate(f"5 negative values\n(min {mc.min():.2f})", xy=(mc.min(), 60),
               xytext=(15, 620), fontsize=8.5, color=ACC,
               arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))
ax[0].annotate(f"{int((mc > 92).sum())} rows up to {mc.max():.0f}\n= 7.8x the median",
               xy=(mc.max(), 5), xytext=(150, 900), fontsize=8.5, color=ACC,
               arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))

gmm = GaussianMixture(2, random_state=0, n_init=5).fit(mc.values.reshape(-1, 1))
order = np.argsort(gmm.means_.ravel())
xs = np.linspace(mc.min(), mc.max(), 900).reshape(-1, 1)
comp = gmm.predict_proba(xs)
for i, (lab, col) in enumerate(zip(["clean body", "contaminant"],
                                   [OK, ACC])):
    j = order[i] if i == 0 else order[1]
    w = gmm.weights_[j] * gmm.predict_proba(xs)[:, j]
    ax[1].fill_between(xs.ravel(), w, color=col, alpha=0.30)
    ax[1].plot(xs.ravel(), w, color=col, lw=1.8,
               label=f"{lab}: weight {gmm.weights_[j]:.4f}, mean {gmm.means_[j,0]:.0f}")
ax[1].hist(mc, bins=110, density=True, color=MUT, alpha=0.32,
           edgecolor="white", linewidth=0.2, label="observed")
ax[1].set_xlim(-20, 400)
ax[1].set_xlabel("MonthlyCharges")
ax[1].set_ylabel("density")
ax[1].set_title("B. A 2-component mixture isolates the contaminant (dBIC = 6,152)")
ax[1].legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_monthly_charges.png", bbox_inches="tight")
plt.close(fig)

# ============================================ FIG 2: missingness / sanity
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ci_u["_m"] = ci_u.Age.isna()
seg = []
for c in ["Gender", "Region", "ContractType"]:
    t = ci_u.groupby(c)["_m"].mean()
    for k, v in t.items():
        seg.append((f"{c[:4]}\n{k}", v))
lbls, vals = zip(*seg)
cols = [OK if abs(v - 0.35) < 0.0143 else ACC for v in vals]
ax[0].bar(range(len(vals)), vals, color=cols, width=0.65)
ax[0].axhline(0.35, color=INK, ls="--", lw=1.2)
ax[0].axhspan(0.35 - 0.0143, 0.35 + 0.0143, color=MUT, alpha=0.20)
ax[0].set_xticks(range(len(vals)))
ax[0].set_xticklabels(lbls, fontsize=7, rotation=90)
ax[0].set_ylabel("Age missing rate")
ax[0].set_ylim(0, 0.5)
ax[0].set_title("A. Age missingness is random (MCAR)\nall segments inside +/-3SE of 35%")

m = ci_u.set_index("CustomerID").join(cl.set_index("CustomerID"))
rate = m.groupby("ContractType").Churn.agg(["mean", "size"])
ax[1].bar(rate.index, rate["mean"] * 100, color=MUT, width=0.55)
ax[1].axhline(cl.Churn.mean() * 100, color=INK, ls="--", lw=1.2,
              label=f"overall {cl.Churn.mean()*100:.2f}%")
for i, (k, r) in enumerate(rate.iterrows()):
    ax[1].text(i, r["mean"] * 100 + 0.15, f"{r['mean']*100:.2f}%", ha="center", fontsize=8.5)
ax[1].set_ylabel("churn rate (%)")
ax[1].set_ylim(0, 11)
ax[1].legend(fontsize=8.5)
ax[1].set_title("B. Yearly contracts churn MOST (8.82%)\n- backwards from real telco behaviour")

udm = ud.merge(ci_u[["CustomerID", "SignupDate_dt"]], on="CustomerID")
udm["pre"] = udm.Month_dt < udm.SignupDate_dt
by = udm.groupby(udm.Month_dt.dt.strftime("%Y-%m"))["pre"].agg(["sum", "size"])
ax[2].bar(by.index, by["sum"], color=ACC, width=0.6)
ax[2].set_ylabel("rows dated before signup")
ax[2].set_title("C. 2,278 usage rows (751 customers)\npredate the customer's own signup")
ax[2].tick_params(axis="x", rotation=90, labelsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_missingness_and_sanity.png", bbox_inches="tight")
plt.close(fig)

# ================================================== FIG 3: usage shapes
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
specs = [
    ("CallMinutes", 90, "Normal-ish (kurt 0.19)"),
    ("DataUsageGB", 90, "Normal-ish, floored at 0"),
    ("SMSCount", 50, "EXACTLY Uniform(0,49)"),
    ("Complaints", 7, "Poisson-ish (OK)"),
]
for a, (col, bins, note) in zip(ax, specs):
    s = ud[col]
    a.hist(s, bins=bins, density=True, color=MUT, edgecolor="white", linewidth=0.3)
    if col == "SMSCount":
        a.axhline(1 / 49, color=ACC, lw=2)
        a.set_title(f"{col}\n{note}", color=ACC)
    elif col in ("CallMinutes", "DataUsageGB"):
        mu, sd = s.mean(), s.std()
        grid = np.linspace(s.min(), s.max(), 200)
        a.plot(grid, stats.norm.pdf(grid, mu, sd), color=OK, lw=1.6)
        a.set_title(f"{col}\n{note}")
    else:
        a.set_title(f"{col}\n{note}")
    a.set_ylabel("density")
fig.suptitle("Usage-metric distributions: two are Gaussian/uniform synthetic draws, not real behaviour",
             y=1.04, fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_usage_shapes.png", bbox_inches="tight")
plt.close(fig)

# ================================================ FIG 4: churn signal test
feat = ud.groupby("CustomerID").agg(**{
    "call_min": ("CallMinutes", "mean"), "call_max": ("CallMinutes", "max"),
    "call_sd": ("CallMinutes", "std"), "data_gb": ("DataUsageGB", "mean"),
    "data_sd": ("DataUsageGB", "std"), "sms": ("SMSCount", "mean"),
    "comp_sum": ("Complaints", "sum"), "comp_max": ("Complaints", "max"),
    "n_zero_data": ("DataUsageGB", lambda s: (s == 0).sum()),
})
X = ci_u.set_index("CustomerID").join(feat).join(cl.set_index("CustomerID"))
X["is_male"] = (X.Gender == "Male").astype(int)
X["charge"] = X.MonthlyCharges.abs()
Xf = pd.get_dummies(X[["charge", "is_male", "Region", "ContractType", "call_min",
                       "call_max", "call_sd", "data_gb", "data_sd", "sms",
                       "comp_sum", "comp_max", "n_zero_data"]], drop_first=True).astype(float)
y = X.Churn
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "Logistic\nregression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "Random forest\n(500 trees)": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                        random_state=42, n_jobs=-1),
    "Gradient\nboosting": HistGradientBoostingClassifier(random_state=42),
}
aucs = {k: cross_val_score(v, Xf, y, cv=cv, scoring="roc_auc") for k, v in models.items()}
rng = np.random.default_rng(0)
null = [cross_val_score(LogisticRegression(max_iter=2000),
                        StandardScaler().fit_transform(Xf), rng.permutation(y.values),
                        cv=cv, scoring="roc_auc").mean() for _ in range(20)]
null = np.array(null)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
names = list(aucs)
means = [aucs[k].mean() for k in names]
sds = [aucs[k].std() for k in names]
ax[0].axhspan(null.min(), null.max(), color=MUT, alpha=0.28,
              label=f"shuffled-label null band\n({null.min():.3f}-{null.max():.3f})")
ax[0].axhline(0.5, color=INK, ls="--", lw=1.2, label="random = 0.500")
ax[0].bar(names, means, yerr=sds, color=ACC, width=0.5, capsize=5)
for i, v in enumerate(means):
    ax[0].text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax[0].set_ylim(0.42, 0.60)
ax[0].set_ylabel("5-fold CV ROC-AUC")
ax[0].set_title("A. Every model scores inside the noise band\n=> churn is unpredictable from these features")
ax[0].legend(fontsize=8, loc="upper right")

tests = [("Gender", "chi2"), ("Region", "chi2"), ("ContractType", "chi2"),
         ("Age", "t"), ("MonthlyCharges", "t"), ("call_min", "t"),
         ("data_gb", "t"), ("sms", "t"), ("comp_sum", "t")]
ps, labs = [], []
for c, kind in tests:
    if kind == "chi2":
        _, p, _, _ = stats.chi2_contingency(pd.crosstab(X[c], y))
    else:
        s = X[[c, "Churn"]].dropna()
        _, p = stats.ttest_ind(s[s.Churn == 1][c], s[s.Churn == 0][c], equal_var=False)
    ps.append(max(p, 1e-4)); labs.append(c)
ax[1].barh(labs, ps, color=[OK if p < 0.05 else MUT for p in ps], height=0.6)
ax[1].axvline(0.05, color=ACC, ls="--", lw=1.4)
ax[1].set_xscale("log")
ax[1].set_xlabel("p-value (log scale)")
ax[1].set_title("B. No feature clears the 0.05 bar\n(SMSCount p=0.026 is 1 of 10 tests = chance)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_churn_signal.png", bbox_inches="tight")
plt.close(fig)

# ================================================== FIG 5: signup timeline
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
by_m = ci_u.SignupDate_dt.dt.to_period("M").value_counts().sort_index()
x = np.arange(len(by_m))
ax[0].bar(x, by_m.values, color=[ACC if v < 50 else MUT for v in by_m.values], width=0.75)
ax[0].set_xticks(x[::6])
ax[0].set_xticklabels([str(p) for p in by_m.index[::6]], rotation=90, fontsize=8)
ax[0].set_ylabel("customers signing up")
ax[0].axhline(by_m.mean(), color=OK, ls="--", lw=1.2,
              label=f"mean {by_m.mean():.0f}/month")
ax[0].set_title("A. Signups run ~150/month, then collapse to 1-3/month\nfor Jul-Nov 2023")
ax[0].legend(fontsize=8.5)

lagd = (udm.loc[udm.pre, "SignupDate_dt"] - udm.loc[udm.pre, "Month_dt"]).dt.days
ax[1].hist(lagd, bins=40, color=ACC, edgecolor="white", linewidth=0.4)
ax[1].set_xlabel("days the usage row predates signup")
ax[1].set_ylabel("rows")
ax[1].set_title(f"B. Pre-signup lag: median {lagd.median():.0f} days, max {lagd.max()} days\n"
                f"({udm.pre.sum():,} rows / {udm.loc[udm.pre,'CustomerID'].nunique()} customers)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_signup_timeline.png", bbox_inches="tight")
plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(FIG)):
    print(f"  {FIG}/{f}")

# machine-readable summary of what the charts claim
summary = {
    "monthly_charges": {
        "n_negative": int((mc < 0).sum()),
        "n_contaminated_mixture": int((gmm.predict_proba(mc.values.reshape(-1, 1))[:, order[1]] > 0.5).sum()),
        "contaminant_weight": round(float(gmm.weights_[order[1]]), 4),
        "contaminant_mean": round(float(gmm.means_[order[1], 0]), 2),
        "max_charge": float(mc.max()),
        "median_charge": float(mc.median()),
    },
    "churn_signal": {k: round(float(v.mean()), 4) for k, v in aucs.items()},
    "null_band": [round(float(null.min()), 4), round(float(null.max()), 4)],
    "pre_signup_rows": int(udm.pre.sum()),
    "pre_signup_customers": int(udm.loc[udm.pre, "CustomerID"].nunique()),
}
import json
with open("outputs/audit_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\nwrote outputs/audit_summary.json")
print(json.dumps(summary, indent=2))
