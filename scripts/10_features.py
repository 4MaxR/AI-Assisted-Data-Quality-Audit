"""Step 1 of the submission: engineer the behavioural feature set.

Applies the documented cleaning rules from the audit (scripts/07) and builds one
row per customer. Every feature is defined in FEATURES below with its rationale,
so the notebook can explain the strategy rather than just list column names.

Key design decision: the brief says churn is driven by a usage DECLINE, so trend
features (slope, first-3 vs last-3 ratio, last-minus-first) are first-class
citizens here, not an afterthought. A standard deviation cannot express direction.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- #
# Feature families (documented rationale)
# ---------------------------------------------------------------------------- #
FEATURES = {
    "demographic": [
        "Age_imputed", "age_missing", "is_male",
    ],
    "contract": [
        "MonthlyCharges_clean", "charge_invalid", "tenure_months",
        "signup_year", "signed_up_in_window",
    ],
    "usage_level": [],       # per-metric mean / median / min / max
    "usage_volatility": [],  # per-metric sd / cv
    "usage_trend": [],       # per-metric slope / half-ratio / last-minus-first
    "engagement": [
        "n_zero_data_months", "n_complaint_months", "complaints_total",
        "complaints_per_call_hour", "data_per_call_min", "sms_per_call_min",
        "zero_data_complaint_interaction",
    ],
}

METRICS = [
    ("CallMinutes", "call"),
    ("DataUsageGB", "data"),
    ("SMSCount", "sms"),
    ("Complaints", "comp"),
]


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    """OLS slope: units of the metric per month. Negative = decline."""
    if len(v) < 2 or np.all(v == v[0]):
        return 0.0
    return float(np.polyfit(t, v, 1)[0])


def _trend_r2(t: np.ndarray, v: np.ndarray) -> float:
    """How consistently the series trends. 0 = pure noise, 1 = perfect line."""
    if len(v) < 2 or np.all(v == v[0]):
        return 0.0
    r = np.corrcoef(t, v)[0, 1]
    return float(r ** 2) if np.isfinite(r) else 0.0


def build_features(verbose: bool = True) -> pd.DataFrame:
    ci = pd.read_csv("data/customer_info.csv")
    ud = pd.read_csv("data/usage_data.csv")
    cl = pd.read_csv("data/churn_labels.csv")

    # ---- cleaning rules carried over from the audit ------------------------
    n_raw = len(ci)
    ci = ci.drop_duplicates(subset="CustomerID", keep="first").copy()
    dup_conflict = pd.read_csv("outputs/clean/customer_info_clean.csv")[
        ["CustomerID", "dq_dup_conflict"]]

    ci["age_missing"] = ci.Age.isna().astype(int)
    ci["Age_imputed"] = ci.Age.fillna(ci.Age.median())
    ci["is_male"] = (ci.Gender == "Male").astype(int)

    clean = pd.read_csv("outputs/clean/customer_info_clean.csv")
    ci = ci.merge(clean[["CustomerID", "MonthlyCharges_clean",
                         "dq_charge_contaminated", "dq_charge_negative",
                         "dq_charge_implausible_low"]], on="CustomerID", how="left")
    ci["charge_invalid"] = (ci.dq_charge_contaminated
                            | ci.dq_charge_negative).astype(int)
    # keep a usable numeric charge even when flagged: fall back to the median of
    # the clean body so no customer is dropped (10k rows, 0.78% affected).
    ci["MonthlyCharges_clean"] = ci.MonthlyCharges_clean.fillna(
        ci.MonthlyCharges_clean.median())

    ci["SignupDate_dt"] = pd.to_datetime(ci.SignupDate, errors="coerce")
    # tenure to the end of the observation window (2023-12-31)
    ci["tenure_months"] = ((pd.Timestamp("2023-12-31") - ci.SignupDate_dt).dt.days
                           / 30.44).round(0)
    ci["signup_year"] = ci.SignupDate_dt.dt.year
    ci["signed_up_in_window"] = (ci.SignupDate_dt >= "2023-01-01").astype(int)

    # ---- usage: level, volatility, trend -----------------------------------
    ud["t"] = pd.to_datetime(ud.Month).dt.month
    ud = ud.sort_values(["CustomerID", "t"])

    rows = []
    for cid, g in ud.groupby("CustomerID", sort=False):
        rec = {"CustomerID": cid}
        t = g.t.to_numpy(float)
        for col, tag in METRICS:
            v = g[col].to_numpy(float)
            rec[f"{tag}_mean"] = v.mean()
            rec[f"{tag}_median"] = np.median(v)
            rec[f"{tag}_min"] = v.min()
            rec[f"{tag}_max"] = v.max()
            rec[f"{tag}_sd"] = v.std()
            rec[f"{tag}_cv"] = v.std() / v.mean() if v.mean() > 0 else 0.0
            # --- the "decline before churn" family ---
            rec[f"{tag}_slope"] = _slope(t, v)
            rec[f"{tag}_rel_slope"] = (_slope(t, v) / v.mean()
                                       if v.mean() > 0 else 0.0)
            rec[f"{tag}_halfratio"] = (v[-3:].mean() / v[:3].mean()
                                       if v[:3].mean() > 0 else np.nan)
            rec[f"{tag}_last_minus_first"] = v[-1] - v[0]
            rec[f"{tag}_trend_r2"] = _trend_r2(t, v)
            rec[f"{tag}_declining"] = int(_slope(t, v) < 0)
        # engagement
        rec["n_zero_data_months"] = int((g.DataUsageGB == 0).sum())
        rec["n_complaint_months"] = int((g.Complaints > 0).sum())
        rec["complaints_total"] = int(g.Complaints.sum())
        rec["call_hours"] = g.CallMinutes.sum() / 60
        rec["complaints_per_call_hour"] = (rec["complaints_total"]
                                           / rec["call_hours"] if rec["call_hours"] else 0)
        rec["data_per_call_min"] = (g.DataUsageGB.sum()
                                    / g.CallMinutes.sum() if g.CallMinutes.sum() else 0)
        rec["sms_per_call_min"] = (g.SMSCount.sum()
                                   / g.CallMinutes.sum() if g.CallMinutes.sum() else 0)
        rec["zero_data_complaint_interaction"] = (rec["n_zero_data_months"]
                                                  * rec["complaints_total"])
        rows.append(rec)

    F = pd.DataFrame(rows).set_index("CustomerID")

    df = (ci.set_index("CustomerID")
            .join(F)
            .join(cl.set_index("CustomerID"))
            .join(dup_conflict.set_index("CustomerID")))

    drop = ["Age", "Gender", "MonthlyCharges", "SignupDate", "SignupDate_dt",
            "dq_charge_contaminated", "dq_charge_negative",
            "dq_charge_implausible_low"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df["dq_dup_conflict"] = df["dq_dup_conflict"].fillna(False).astype(int)

    if verbose:
        print(f"customer_info rows : {n_raw:,} -> {len(ci):,} (deduplicated)")
        print(f"feature table      : {df.shape[0]:,} customers x {df.shape[1]} columns")
        print(f"target             : Churn, {df.Churn.sum():.0f} positives "
              f"({df.Churn.mean():.4%})")
        print(f"nulls              : {int(df.isna().sum().sum())}")
    return df


if __name__ == "__main__":
    df = build_features()
    df.to_csv("outputs/features.csv")

    num = df.select_dtypes(include=[np.number]).columns
    fam = pd.DataFrame({
        "family": ["demographic", "contract", "usage_level", "usage_volatility",
                   "usage_trend", "engagement"],
        "example": ["Age_imputed, age_missing, is_male",
                    "MonthlyCharges_clean, tenure_months, charge_invalid",
                    "call_mean, data_median, sms_max, comp_min",
                    "call_sd, data_cv, sms_sd, comp_cv",
                    "call_slope, data_halfratio, sms_rel_slope, comp_last_minus_first",
                    "n_complaint_months, data_per_call_min, ..."],
    })
    print("\nfeature families:")
    print(fam.to_string(index=False))
    print(f"\ntotal numeric features: {len(num)}")
    print("\ncolumns:")
    print(", ".join(df.columns))
    print("\nwrote outputs/features.csv")
