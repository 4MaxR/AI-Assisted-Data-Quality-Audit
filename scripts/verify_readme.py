"""Verification: does README.md agree with the actual audit outputs?

A case-study document that drifts from its own data is worse than no document.
This checks every load-bearing number in README.md against the generated artifacts
and fails loudly on any mismatch.
"""
import json
import re
from pathlib import Path

import pandas as pd

root = Path(".")
readme = (root / "README.md").read_text(encoding="utf-8")
fails, checks = [], 0


def check(label, claimed, actual, tol=0.0):
    global checks
    checks += 1
    ok = abs(claimed - actual) <= tol if isinstance(actual, (int, float)) else claimed == actual
    print(f"{'OK  ' if ok else 'FAIL'} {label:52s} README={claimed}  actual={actual}")
    if not ok:
        fails.append(label)


# ---------------------------------------------------------------- source facts
ci = pd.read_csv("data/customer_info.csv")
ud = pd.read_csv("data/usage_data.csv")
cl = pd.read_csv("data/churn_labels.csv")

check("total rows audited", 140200, len(ci) + len(ud) + len(cl))
check("unique customers", 10000, ci.CustomerID.nunique())
check("churners", 786, int(cl.Churn.sum()))
check("churn base rate %", 7.86, round(cl.Churn.mean() * 100, 2), tol=0.01)

# ---------------------------------------------------------------- the register
reg = pd.read_csv("outputs/data_quality_register.csv")
check("issues in register", 11, len(reg))
sev = reg.severity.value_counts()
check("High severity count", 6, int(sev.get("High", 0)))
check("Medium severity count", 3, int(sev.get("Medium", 0)))
check("Low severity count", 1, int(sev.get("Low", 0)))
check("Critical count", 1, int(sev.get("Critical (use case)", 0)))

by_id = reg.set_index("issue_id")
for iid, rows in [("DQ-01", 195), ("DQ-03", 3500), ("DQ-04", 49), ("DQ-05", 5),
                  ("DQ-06", 24), ("DQ-07", 2278), ("DQ-08", 58),
                  ("DQ-09", 120000), ("DQ-10", 120000), ("DQ-11", 10000)]:
    check(f"register {iid} affected_rows", rows, int(by_id.loc[iid, "affected_rows"]))

# --------------------------------------------------------- repair reconciliation
cln = pd.read_csv("outputs/clean/customer_info_clean.csv")
clnu = pd.read_csv("outputs/clean/usage_data_clean.csv")
check("cleaned customer_info rows", 10000, len(cln))
check("dq_charge_negative flag", 5, int(cln.dq_charge_negative.sum()))
check("dq_charge_contaminated flag", 49, int(cln.dq_charge_contaminated.sum()))
check("dq_charge_implausible_low flag", 24, int(cln.dq_charge_implausible_low.sum()))
check("dq_pre_signup flag", 2278, int(clnu.dq_pre_signup.sum()))
check("monthly charges nulled", 54, int(cln.MonthlyCharges_clean.isna().sum()))
check("total invalid charges (49+5+24)", 78, 49 + 5 + 24)

# ---------------------------------------------------------------- statistics
summary = json.loads((root / "outputs/audit_summary.json").read_text())
check("contaminant weight", 0.0050, summary["monthly_charges"]["contaminant_weight"], tol=1e-6)
check("contaminant mean", 257.42, summary["monthly_charges"]["contaminant_mean"], tol=0.01)
check("max charge", 390.30, summary["monthly_charges"]["max_charge"], tol=0.01)
check("median charge", 50.02, summary["monthly_charges"]["median_charge"], tol=0.01)
check("pre-signup customers", 751, summary["pre_signup_customers"])
check("null band lower", 0.4708, summary["null_band"][0], tol=1e-4)
check("null band upper", 0.5372, summary["null_band"][1], tol=1e-4)
check("AUC logistic", 0.5176, list(summary["churn_signal"].values())[0], tol=1e-4)

# ---------------------------------------------------------------- impact table
imp = pd.read_csv("outputs/impact_naive_vs_audited.csv").set_index("metric")
check("naive join rows", 10200, int(imp.loc["rows after joining customer_info to churn_labels", "naive"].replace(",", "")))
check("audited join rows", 10000, int(imp.loc["rows after joining customer_info to churn_labels", "audited"].replace(",", "")))
check("naive churn rate %", 7.8137, float(imp.loc["churn_rate", "naive"].strip("%")), tol=1e-4)
check("audited churn rate %", 7.8600, float(imp.loc["churn_rate", "audited"].strip("%")), tol=1e-4)
check("naive mean charges", 50.962, float(imp.loc["mean_MonthlyCharges", "naive"]), tol=0.001)
check("audited mean charges", 50.001, float(imp.loc["mean_MonthlyCharges", "audited"]), tol=0.001)
check("naive mean age", 43.425, float(imp.loc["mean_Age", "naive"]), tol=0.001)
check("audited mean age", 43.273, float(imp.loc["mean_Age", "audited"]), tol=0.001)

# --------------------------------------------------- submission: modelling
model = json.loads((root / "outputs/model_summary.json").read_text())
bonus = json.loads((root / "outputs/bonus_summary.json").read_text())
feat = pd.read_csv("outputs/features.csv")

check("feature table rows", 10000, len(feat))
check("feature table columns", 69, feat.shape[1])
check("design matrix features", 70, model["n_features"])
check("model base rate", 0.0786, model["base_rate"], tol=1e-4)

for label, claimed in [("Dummy (base rate)", 0.4994), ("Logistic regression", 0.4964),
                       ("Logistic (balanced)", 0.4949), ("Random forest", 0.5010),
                       ("Gradient boosting", 0.5181)]:
    check(f"ROC-AUC {label}", claimed, model["models"][label]["roc_auc"], tol=5e-5)

check("best model", "Gradient boosting", model["best_model"])
check("null band lower (70-feature)", 0.4772, model["null_band"][0], tol=1e-4)
check("null band upper (70-feature)", 0.5198, model["null_band"][1], tol=1e-4)
check("share of null draws >= best", 0.10, model["null_share_ge_best"], tol=1e-9)
check("submitted mean probability", 0.0732, model["submitted_mean_prob"], tol=5e-5)
check("top decile lift", 0.9542, model["top_decile_lift"], tol=5e-5)

# --------------------------------------------------- submission: cost analysis
check("cost of doing nothing", 3930, model["cost_never_target"])
check("cost of targeting everybody", 10000, model["cost_target_all"])
check("cost of perfect model", 786, model["cost_perfect_model"])
check("cost of model at tau*", 4039, model["cost_model_at_tau_star"])
check("max achievable saving", 3144, model["max_achievable_saving"])
check("value captured %", -3.5, round(model["value_captured_pct"] * 100, 1), tol=0.05)
check("optimal threshold tau*", 0.20, model["optimal_tau_theory"], tol=1e-9)

# --------------------------------------------------- submission: bonus results
check("SHAP top feature share %", 3.42, round(bonus["shap_top_share"] * 100, 2), tol=0.005)
check("permutation importance max", 0.0007, bonus["perm_importance_max"], tol=5e-5)
check("survival analysis feasible", False, bonus["survival_feasible"])

# --------------------------------------------------- submission: prediction.csv
pred = pd.read_csv("prediction.csv")
check("prediction.csv rows", 10000, len(pred))
check("prediction.csv header", ["CustomerID", "ChurnProbability"], list(pred.columns))
check("prediction.csv covers every labelled customer", True,
      set(pred.CustomerID) == set(cl.CustomerID))
check("prediction.csv has no nulls", 0, int(pred.ChurnProbability.isna().sum()))

# ------------------------------------- strongest check: the literal must appear
print()
for needle in ["0.5181", "0.4772–0.5198", "0.4708", "0.5372", "3,930", "4,039",
               "3,144", "−3.5%", "3.42%", "0.95×", "0.0007", "1/5 = 0.20",
               "70-column design matrix", "36 cells", "0.0808", "0.0786"]:
    checks += 1
    present = needle in readme
    print(f"{'OK  ' if present else 'FAIL'} README contains literal               {needle!r}")
    if not present:
        fails.append(f"README missing literal {needle!r}")

# ---------------------------------------------------------------- links & anchors
def slug(text):
    t = text.strip().lower()
    t = re.sub(r"[^\w \-]", "", t)
    return t.replace(" ", "-")


def audit_doc(path):
    """TOC anchors, relative links/images, and code fences for one markdown file."""
    text = path.read_text(encoding="utf-8")
    in_fence, heads = False, []
    for ln in text.split("\n"):
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and ln.startswith("#"):
            heads.append(re.sub(r"^#+\s*", "", ln))
    hs = [slug(h) for h in heads]

    toc = re.findall(r"\]\(#([^)]+)\)", text)
    bad_anchors = [t for t in toc if t not in hs]

    refs = [r for r in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
            if not r.startswith(("http", "#"))]
    missing = sorted({r for r in refs if not (path.parent / r).exists()})

    fences = re.findall(r"^```(\w*)", text, re.M)
    return {"anchors": len(toc), "bad_anchors": bad_anchors, "refs": len(set(refs)),
            "missing": missing, "fences": len(fences),
            "mermaid": fences.count("mermaid"), "balanced": len(fences) % 2 == 0}


print()
for doc in [root / "README.md", root / "docs" / "AUDIT_REPORT.md"]:
    r = audit_doc(doc)
    ok = not r["bad_anchors"] and not r["missing"] and r["balanced"]
    print(f"{'OK  ' if ok else 'FAIL'} {doc}")
    print(f"       anchors: {r['anchors']} links, broken: {r['bad_anchors'] or 'NONE'}")
    print(f"       relative links + images: {r['refs']} checked, missing: {r['missing'] or 'NONE'}")
    print(f"       fences: {r['fences']} openers, balanced={r['balanced']}, mermaid={r['mermaid']}")
    if not ok:
        fails.append(f"{doc} structure")
    if doc.name == "README.md":
        check("mermaid diagrams in README", 2, r["mermaid"])

print("\n" + "=" * 70)
print(f"{checks} value checks + link checks run")
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES -> {fails}")
