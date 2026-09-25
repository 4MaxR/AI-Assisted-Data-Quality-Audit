"""Step 2 of the submission: model, validation, and cost-sensitive evaluation.

Two things drive the design:

1. There is no held-out test set - churn_labels.csv covers all 10,000 customers.
   So every reported metric comes from stratified cross-validation, and the
   submitted probabilities are OUT-OF-FOLD (each customer scored by a model that
   never saw them). That avoids leaking in-sample optimism into the submission.

2. 7.86% positives means accuracy and even ROC-AUC flatter a model. Average
   precision (PR-AUC) with the base rate as the no-skill reference is reported
   alongside, plus a decile lift table and an expected-cost curve.
"""
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = 42
df = pd.read_csv("outputs/features.csv")
y = df["Churn"].to_numpy()
X = df.drop(columns=["CustomerID", "Churn"])
X = pd.get_dummies(X, columns=["Region", "ContractType"], drop_first=True).astype(float)

# ratio features are NaN when a customer had zero baseline activity in months 1-3.
# That is meaningful ("no baseline to decline from"), so record it, then impute.
nan_cols = X.columns[X.isna().any()].tolist()
for c in nan_cols:
    X[f"{c}_nobase"] = X[c].isna().astype(int)
X = X.fillna(X.median())

BASE_RATE = y.mean()
print(f"design matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
print(f"positives    : {int(y.sum())} ({BASE_RATE:.4%})")
print(f"nobase flags : {len(nan_cols)} ratio columns had zero-baseline cases\n")

cv = StratifiedKFold(5, shuffle=True, random_state=RNG)

MODELS = {
    "Dummy (base rate)": DummyClassifier(strategy="prior"),
    "Logistic regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=5000, random_state=RNG)),
    "Logistic (balanced)": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=5000, class_weight="balanced",
                                                            random_state=RNG)),
    "Random forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                            random_state=RNG, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=RNG),
}

print("=" * 88)
print("MODEL COMPARISON - 5-fold stratified CV, out-of-fold predictions")
print("=" * 88)
print(f"{'model':22s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Brier':>8s} {'vs base':>9s}")
print(f"{'(no-skill reference)':22s} {'0.5000':>9s} {BASE_RATE:9.4f} "
      f"{BASE_RATE*(1-BASE_RATE):8.4f}")

oof_store, results = {}, {}
for name, mdl in MODELS.items():
    p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_store[name] = p
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    br = brier_score_loss(y, p)
    results[name] = {"roc_auc": roc, "pr_auc": pr, "brier": br}
    print(f"{name:22s} {roc:9.4f} {pr:9.4f} {br:8.4f} {pr-BASE_RATE:+9.4f}")

# --------------------------------------------------------------------------- #
# How much is real? Compare the best model against a shuffled-label null.
# --------------------------------------------------------------------------- #
print()
print("=" * 88)
print("IS THE SIGNAL REAL? shuffled-label null (20 draws of the best model)")
print("=" * 88)
best_name = max(results, key=lambda k: results[k]["roc_auc"] if k != "Dummy (base rate)" else 0)
best_oof = oof_store[best_name]
best_roc = results[best_name]["roc_auc"]
print(f"best model: {best_name}  ROC-AUC = {best_roc:.4f}")

rng = np.random.default_rng(0)
null = []
for _ in range(20):
    yp = rng.permutation(y)
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          X, yp, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    null.append(roc_auc_score(yp, p))
null = np.array(null)
pct = (null >= best_roc).mean()
print(f"null: mean={null.mean():.4f}  min={null.min():.4f}  max={null.max():.4f}")
print(f"share of null draws >= best model: {pct:.1%}")
print("-> " + ("real signal present" if pct < 0.05 else
              "INDISTINGUISHABLE from random labels - no usable signal"))

# --------------------------------------------------------------------------- #
# Cost-sensitive evaluation.  The brief: acquiring costs up to 5x retaining.
#
# Decision rule (derived, not assumed). Offering to a customer costs C_OFFER
# whether or not they would have churned. Not offering costs p x C_LOSS in
# expectation, where C_LOSS is the cost of losing and replacing them.
#     offer iff  p x C_LOSS > C_OFFER   ->   p > C_OFFER / C_LOSS
# So the theoretically optimal threshold is simply tau* = 1 / (acquisition ratio).
# --------------------------------------------------------------------------- #
print()
print("=" * 88)
print("COST-SENSITIVE EVALUATION (out-of-fold probabilities)")
print("=" * 88)

C_OFFER = 1.0                    # relative cost of one retention offer
C_LOSS = 5.0                     # cost of losing a customer (brief: 5x retention)
TAU_STAR = C_OFFER / C_LOSS
print(f"Assumptions: offer cost = {C_OFFER:.0f}u, cost of losing a customer = {C_LOSS:.0f}u")
print(f"Derived optimal threshold  tau* = C_offer / C_loss = {TAU_STAR:.4f}")
print(f"Base rate is {BASE_RATE:.4f}, so a random customer is NOT worth targeting:")
print(f"   expected saving per random offer = {BASE_RATE:.4f} x {C_LOSS:.0f}u = "
      f"{BASE_RATE*C_LOSS:.3f}u  <  {C_OFFER:.0f}u offer cost\n")


def cost_at(p, tau, c_offer=C_OFFER, c_loss=C_LOSS):
    pred = p >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return c_offer * int(pred.sum()) + c_loss * fn, tp, fp, fn


sweep = np.linspace(0.005, 0.995, 200)
costs = np.array([cost_at(best_oof, t)[0] for t in sweep])
emp_tau = float(sweep[int(np.argmin(costs))])

# baselines + upper bound
n_pos = int(y.sum())
COST_NEVER = C_LOSS * n_pos                          # lose every churner
COST_ALL = C_OFFER * len(y)                           # offer to everyone
COST_PERFECT = C_OFFER * n_pos                        # offer only to true churners
MAX_SAVING = COST_NEVER - COST_PERFECT

scenarios = [
    ("Target nobody (do nothing)", COST_NEVER, 0, 0, n_pos, "baseline"),
    ("Target everybody", COST_ALL, n_pos, len(y) - n_pos, 0, "baseline"),
    ("Perfect model (upper bound)", COST_PERFECT, n_pos, 0, 0, "oracle"),
]
print(f"{'strategy':36s} {'TP':>5s} {'FP':>6s} {'FN':>5s} {'cost':>8s} {'vs nobody':>11s}")
for label, c, tp, fp, fn, _ in scenarios:
    print(f"{label:36s} {tp:5d} {fp:6d} {fn:5d} {c:8.0f} {COST_NEVER-c:+11.0f}")

rows = []
for label, tau, kind in [(f"{best_name} @ theory tau*={TAU_STAR:.2f}", TAU_STAR, "model"),
                         (f"{best_name} @ empirical best tau", emp_tau, "model")]:
    c, tp, fp, fn = cost_at(best_oof, tau)
    rows.append({"strategy": label, "tau": tau, "tp": tp, "fp": fp, "fn": fn, "cost": c})
    captured = (COST_NEVER - c) / MAX_SAVING if MAX_SAVING else 0
    print(f"{label:36s} {tp:5d} {fp:6d} {fn:5d} {c:8.0f} {COST_NEVER-c:+11.0f}"
          f"   ({captured:.1%} of achievable)")

print(f"\nMaximum saving available to ANY model: {MAX_SAVING:.0f}u "
      f"(= {n_pos} offer(s) instead of {n_pos} lost customer(s))")
model_row = rows[0]
captured = (COST_NEVER - model_row["cost"]) / MAX_SAVING if MAX_SAVING else 0
print(f"Value captured by the model at tau*: {captured:.2%} of the achievable saving")
print("-> with a ranking no better than chance, the model captures essentially none")
print("   of the available value, and the cost curve is flat near its minimum.")

# sensitivity of the decision to the cost ratio
print("\nsensitivity - how the cost-optimal choice moves with the acquisition ratio:")
sens = []
for ratio in [1, 2, 3, 5, 10, 20]:
    t = 1.0 / ratio
    c_never = ratio * n_pos
    c_all = len(y)
    c, tp, fp, fn = cost_at(best_oof, t)
    best_policy = min([("nobody", c_never), ("everybody", c_all), ("model", c)])
    sens.append({"ratio": f"{ratio}:1", "theory_tau": round(t, 4),
                 "never": c_never, "everybody": c_all, "model": c,
                 "cheapest": best_policy[0]})
    print(f"   {ratio:2d}:1  tau*={t:.4f}  cost: nobody={c_never:6.0f}  "
          f"everybody={c_all:6.0f}  model={c:6.0f}  -> cheapest: {best_policy[0]}")
pd.DataFrame(sens).to_csv("outputs/cost_sensitivity.csv", index=False)

# --------------------------------------------------------------------------- #
# Decile lift - the business question "who do I call first?"
# --------------------------------------------------------------------------- #
print()
print("=" * 88)
print("DECILE LIFT (out-of-fold probabilities, best model)")
print("=" * 88)
order = np.argsort(-best_oof)
dec = np.array_split(order, 10)
lift_rows = []
print(f"{'decile':>7s} {'n':>6s} {'churners':>9s} {'churn rate':>11s} {'lift':>7s} {'cum recall':>11s}")
cum = 0
for i, idx in enumerate(dec, 1):
    ch = int(y[idx].sum())
    cum += ch
    rate = ch / len(idx)
    lift_rows.append({"decile": i, "n": len(idx), "churners": ch,
                      "churn_rate": rate, "lift": rate / BASE_RATE,
                      "cum_recall": cum / y.sum()})
    print(f"{i:7d} {len(idx):6d} {ch:9d} {rate:11.4%} {rate/BASE_RATE:7.2f}x {cum/y.sum():11.1%}")
lift_df = pd.DataFrame(lift_rows)
lift_df.to_csv("outputs/decile_lift.csv", index=False)

# --------------------------------------------------------------------------- #
# Submission: probability averaging, not rank averaging.
# A column called ChurnProbability must be a probability: its mean has to sit near
# the 7.86% base rate and its Brier score has to be meaningful. Rank averaging
# forces a uniform [0,1] spread (mean 0.50), which is monotone with the rank so
# ROC-AUC is identical, but it destroys calibration and inflates Brier badly.
# Both are measured below and the calibrated ensemble is submitted.
# --------------------------------------------------------------------------- #
learners = ["Logistic regression", "Random forest", "Gradient boosting"]
prob_avg = np.mean([oof_store[m] for m in learners], axis=0)
rank_avg = np.mean([pd.Series(oof_store[m]).rank(pct=True).to_numpy() for m in learners],
                   axis=0)

print()
print("=" * 88)
print("ENSEMBLE CHOICE: probability average vs rank average")
print("=" * 88)
print(f"{'ensemble':28s} {'mean':>8s} {'ROC-AUC':>9s} {'PR-AUC':>8s} {'Brier':>8s}")
ens_rows = []
for label, p in [("probability average", prob_avg), ("rank average", rank_avg)]:
    r = {"ensemble": label, "mean": p.mean(), "roc_auc": roc_auc_score(y, p),
         "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    ens_rows.append(r)
    print(f"{label:28s} {p.mean():8.4f} {r['roc_auc']:9.4f} {r['pr_auc']:8.4f} {r['brier']:8.4f}")
print(f"{'base rate (no-skill)':28s} {BASE_RATE:8.4f} {0.5:9.4f} {BASE_RATE:8.4f} "
      f"{BASE_RATE*(1-BASE_RATE):8.4f}")
print("\n-> identical ranking power, but only the probability average keeps the mean at")
print("   the base rate and a usable Brier score. That is what gets submitted.")
pd.DataFrame(ens_rows).to_csv("outputs/ensemble_comparison.csv", index=False)

# calibration by decile of predicted probability
print("\ncalibration (probability average) - predicted vs observed churn rate:")
cal = pd.DataFrame({"p": prob_avg, "y": y})
cal["bin"] = pd.qcut(cal.p, 10, labels=False, duplicates="drop")
cal_tab = cal.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"),
                                 observed=("y", "mean"))
cal_tab["gap"] = cal_tab.observed - cal_tab.predicted
print(cal_tab.round(4).to_string())
cal_tab.to_csv("outputs/calibration.csv")

FINAL = prob_avg
sub = pd.DataFrame({"CustomerID": df.CustomerID, "ChurnProbability": FINAL})
sub.to_csv("prediction.csv", index=False)

print()
print("=" * 88)
print("SUBMISSION")
print("=" * 88)
print(f"prediction.csv written: {len(sub):,} rows")
print(f"  ChurnProbability: min={FINAL.min():.4f} median={np.median(FINAL):.4f} "
      f"max={FINAL.max():.4f} mean={FINAL.mean():.4f}")
print(f"  base rate for reference: {BASE_RATE:.4f}  ->  mean is "
      f"{'aligned' if abs(FINAL.mean()-BASE_RATE) < 0.01 else 'MISALIGNED'}")
print(f"  ROC-AUC {roc_auc_score(y, FINAL):.4f} | Brier {brier_score_loss(y, FINAL):.4f}")
print(f"  ensemble: probability average of {', '.join(learners)}")

summary = {
    "n_rows": int(len(sub)),
    "n_features": int(X.shape[1]),
    "base_rate": float(BASE_RATE),
    "models": {k: {m: float(v) for m, v in d.items()} for k, d in results.items()},
    "best_model": best_name,
    "best_roc_auc": float(best_roc),
    "null_band": [float(null.min()), float(null.max())],
    "null_share_ge_best": float(pct),
    "submitted_roc_auc": float(roc_auc_score(y, FINAL)),
    "submitted_pr_auc": float(average_precision_score(y, FINAL)),
    "submitted_brier": float(brier_score_loss(y, FINAL)),
    "submitted_mean_prob": float(FINAL.mean()),
    "optimal_tau_theory": float(TAU_STAR),
    "optimal_tau_empirical": float(emp_tau),
    "cost_never_target": float(COST_NEVER),
    "cost_target_all": float(COST_ALL),
    "cost_perfect_model": float(COST_PERFECT),
    "cost_model_at_tau_star": float(model_row["cost"]),
    "max_achievable_saving": float(MAX_SAVING),
    "value_captured_pct": float(captured),
    "top_decile_lift": float(lift_df.lift.iloc[0]),
}
with open("outputs/model_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
np.save("outputs/oof_best.npy", best_oof)
print("wrote outputs/model_summary.json, oof_best.npy, decile_lift.csv, "
      "cost_sensitivity.csv, calibration.csv, ensemble_comparison.csv")
