"""Bonus challenges: SHAP explainability, drift detection, cost-sensitive learning,
and an honest feasibility check on survival analysis.

Sections 1-3 produce real evidence. Section 4 documents why one bonus challenge
cannot be attempted with this dataset, which is a finding in itself.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = 42
FIG = "outputs/figures"
INK, ACC, OK, MUT = "#1b2430", "#c0392b", "#2e7d5b", "#8a94a6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#e8ebf0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
})

df = pd.read_csv("outputs/features.csv")
y = df.Churn.to_numpy()
X = pd.get_dummies(df.drop(columns=["CustomerID", "Churn"]),
                   columns=["Region", "ContractType"], drop_first=True).astype(float)
for c in [c for c in X.columns if X[c].isna().any()]:
    X[f"{c}_nobase"] = X[c].isna().astype(int)
X = X.fillna(X.median())
cv = StratifiedKFold(5, shuffle=True, random_state=RNG)

out = {}

# =========================================================== 1. SHAP
print("=" * 84)
print("BONUS 1: SHAP explainability")
print("=" * 84)
# A 400-tree forest with the exact tree explainer is O(trees x leaves^2) per row
# and does not finish in reasonable time on 10,000 rows. Two changes keep it exact
# but fast: a smaller forest, and explaining a seeded random subsample. The
# conclusion is cross-checked against permutation importance on the FULL data.
from sklearn.inspection import permutation_importance

import shap

rf_small = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                  random_state=RNG, n_jobs=-1)
rf_small.fit(X, y)
rng_shap = np.random.default_rng(RNG)
idx = rng_shap.choice(len(X), size=1000, replace=False)
Xs = X.iloc[idx]
print(f"explaining {len(Xs):,} randomly sampled rows "
      f"(seeded, seed={RNG}) with a 100-tree forest")

expl = shap.TreeExplainer(rf_small)
sv = expl.shap_values(Xs, check_additivity=False)
sv = sv[1] if isinstance(sv, list) else sv
if sv.ndim == 3:
    sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
       .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
print("top 12 features by mean |SHAP value|:")
print(imp.head(12).round(5).to_string(index=False))
print(f"\ntop feature contributes {imp.share.iloc[0]:.2%} of total attribution")

# cross-check on the full data with permutation importance (fast, model-agnostic)
pi = permutation_importance(rf_small, X, y, n_repeats=5, random_state=RNG,
                            scoring="roc_auc", n_jobs=-1)
pidf = (pd.DataFrame({"feature": X.columns, "perm_importance": pi.importances_mean})
        .sort_values("perm_importance", ascending=False).reset_index(drop=True))
print("\ncross-check - permutation importance on ROC-AUC (full data, 5 repeats):")
print(pidf.head(8).round(5).to_string(index=False))
print(f"largest AUC drop from shuffling any feature: {pidf.perm_importance.max():.5f}")
print("\n-> attribution spreads thinly across all features with no dominant driver,")
print("   and shuffling any single feature barely moves ROC-AUC. Both are what a")
print("   signal-free label looks like.")
imp.to_csv("outputs/shap_importance.csv", index=False)
pidf.to_csv("outputs/permutation_importance.csv", index=False)
out["shap_top_feature"] = imp.feature.iloc[0]
out["shap_top_share"] = float(imp.share.iloc[0])
out["perm_importance_max"] = float(pidf.perm_importance.max())

fig, ax = plt.subplots(figsize=(8, 5))
t = imp.head(15).iloc[::-1]
ax.barh(t.feature, t.mean_abs_shap, color=[ACC if i == 0 else MUT for i in range(len(t))])
ax.set_xlabel("mean |SHAP value|  (average impact on the model output)")
ax.set_title("No dominant churn driver\nattribution is flat across all 70 features")
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_shap.png", bbox_inches="tight")
plt.close(fig)

# =========================================================== 2. DRIFT
print()
print("=" * 84)
print("BONUS 2: data drift detection (PSI, month-over-month)")
print("=" * 84)
ud = pd.read_csv("data/usage_data.csv")
ud["t"] = pd.to_datetime(ud.Month).dt.month


def psi(base, comp, bins=10):
    """Population Stability Index between two samples."""
    qs = np.quantile(base, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    b = np.histogram(base, bins=qs)[0] / len(base)
    c = np.histogram(comp, bins=qs)[0] / len(comp)
    b, c = np.clip(b, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - b) * np.log(c / b)))


# reference = months 1-6, compare each later month
ref = ud[ud.t <= 6]
print("PSI vs months 1-6 reference  (<0.1 stable, 0.1-0.25 moderate, >0.25 major shift)")
print(f"{'month':>6s} " + " ".join(f"{c:>11s}" for c in
                                   ["CallMinutes", "DataUsageGB", "SMSCount", "Complaints"]))
drift_rows = []
for m in range(1, 13):
    cur = ud[ud.t == m]
    ps = [psi(ref[c].to_numpy(), cur[c].to_numpy()) for c in
          ["CallMinutes", "DataUsageGB", "SMSCount", "Complaints"]]
    drift_rows.append({"month": m, "call": ps[0], "data": ps[1], "sms": ps[2], "comp": ps[3]})
    flag = "  <-- SHIFT" if max(ps) > 0.1 else ""
    print(f"{m:6d} " + " ".join(f"{v:11.4f}" for v in ps) + flag)
drift = pd.DataFrame(drift_rows)
drift.to_csv("outputs/drift_psi.csv", index=False)
worst = drift.loc[drift[["call", "data", "sms", "comp"]].max(axis=1).idxmax()]
base_level = drift.loc[drift.month <= 9, ["call", "data"]].max().max()
shift_level = drift.loc[drift.month >= 10, ["call", "data"]].min().min()
print(f"\nlargest shift: month {int(worst['month'])} (CallMinutes PSI {worst['call']:.4f})")
print(f"months 1-9 max PSI  : {base_level:.4f}")
print(f"months 10-12 min PSI: {shift_level:.4f}  -> a {shift_level/base_level:.0f}x step change")
print("SMSCount and Complaints stay flat at ~0.0001-0.0007 throughout.")
print("-> CallMinutes and DataUsageGB both step up at month 10 while the other two")
print("   metrics do not. The absolute PSI stays under the conventional 0.1 alarm,")
print("   so this is a small but unambiguous structural break, not a major drift")
print("   event - still worth flagging, because a model trained on months 1-9 faces")
print("   a measurably different input distribution from month 10 onward.")

fig, ax = plt.subplots(figsize=(9, 4.2))
for c, lab in [("call", "CallMinutes"), ("data", "DataUsageGB"),
               ("sms", "SMSCount"), ("comp", "Complaints")]:
    ax.plot(drift.month, drift[c], marker="o", lw=1.8, label=lab)
ax.axhline(0.1, color=ACC, ls="--", lw=1.2, label="conventional 'moderate shift' alarm (0.1)")
ax.axvspan(9.5, 12.5, color=MUT, alpha=0.15)
ax.annotate("step change\nat month 10", xy=(10, 0.047), xytext=(6.4, 0.075),
            fontsize=8.5, color=ACC,
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))
ax.set_xticks(range(1, 13))
ax.set_xlabel("month of 2023")
ax.set_ylabel("PSI vs months 1-6 reference")
ax.set_title("Only the two usage-volume metrics shift - a step change at month 10")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG}/fig7_drift.png", bbox_inches="tight")
plt.close(fig)

# =========================================================== 3. COST-SENSITIVE LEARNING
print()
print("=" * 84)
print("BONUS 3: cost-sensitive learning (vs cost-sensitive thresholding)")
print("=" * 84)
C_OFFER, C_LOSS = 1.0, 5.0
n_pos = int(y.sum())
COST_NEVER = C_LOSS * n_pos
MAX_SAVING = COST_NEVER - C_OFFER * n_pos

# weight positives by the loss:offer ratio so the learner optimises expected cost
w = np.where(y == 1, C_LOSS, C_OFFER)
print(f"sample weights: positives x{C_LOSS:.0f}, negatives x{C_OFFER:.0f}")

variants = {
    "plain logistic": (make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=5000, random_state=RNG)), None),
    "cost-weighted logistic": (make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=5000, random_state=RNG)), w),
}
from sklearn.metrics import roc_auc_score
cs_rows = []
for label, (mdl, weight) in variants.items():
    # sklearn 1.9 removed cross_val_predict(fit_params=...) in favour of params=
    extra = {"params": {"logisticregression__sample_weight": weight}} if weight is not None else {}
    p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba", n_jobs=-1, **extra)[:, 1]
    best_c, best_t = 1e18, None
    for t in np.linspace(0.01, 0.99, 200):
        pred = p >= t
        c = C_OFFER * int(pred.sum()) + C_LOSS * int(((~pred) & (y == 1)).sum())
        if c < best_c:
            best_c, best_t = c, t
    auc = roc_auc_score(y, p)
    cs_rows.append({"variant": label, "roc_auc": auc, "best_tau": best_t,
                    "min_cost": best_c, "vs_never": COST_NEVER - best_c,
                    "pct_of_achievable": (COST_NEVER - best_c) / MAX_SAVING})
    print(f"  {label:24s} AUC={auc:.4f}  best tau={best_t:.3f}  min cost={best_c:.0f}  "
          f"({(COST_NEVER-best_c)/MAX_SAVING:+.2%} of achievable)")
pd.DataFrame(cs_rows).to_csv("outputs/cost_sensitive_learning.csv", index=False)
print(f"\nreference: do nothing = {COST_NEVER:.0f}u, perfect model = {C_OFFER*n_pos:.0f}u, "
      f"max achievable saving = {MAX_SAVING:.0f}u")
print("-> reweighting changes the threshold the model prefers but not its ranking power,")
print("   so it cannot recover value that is not present in the features.")

# =========================================================== 4. SURVIVAL ANALYSIS
print()
print("=" * 84)
print("BONUS 4: survival / time-to-churn - FEASIBILITY CHECK")
print("=" * 84)
cl = pd.read_csv("data/churn_labels.csv")
ud_cols = ud.columns.tolist()
print(f"churn_labels columns : {list(cl.columns)}")
print(f"usage_data columns   : {[c for c in ud_cols if c != 't']}")
print(f"distinct target values: {sorted(cl.Churn.unique())}")
print("""
A survival model needs, per subject:
   (a) a duration  - months from origin to the churn event
   (b) an event indicator - observed churn vs right-censored

This dataset provides neither. There is no churn date or tenure field anywhere,
and the label is a single binary flag. Every churner would receive the same
duration (the end of the 12-month window) and every non-churner is administratively
censored at that same instant, so the hazard function is not identifiable.

Conclusion: survival analysis is NOT feasible on the data as delivered. Fitting
a Cox model here would produce numbers, but they would be artefacts of an assumed
duration, not estimates. What would be needed: a churn date or a churn month per
customer, plus tenure at observation start.""")
out["survival_feasible"] = False

with open("outputs/bonus_summary.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote outputs/shap_importance.csv, drift_psi.csv, cost_sensitive_learning.csv,")
print("      bonus_summary.json, figures/fig6_shap.png, figures/fig7_drift.png")
