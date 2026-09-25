"""Pass 1: structural profiling of the three raw tables.

Answers: what is the grain, what dtypes did pandas infer, how much is missing,
and what are the raw value ranges? No cleaning, no assumptions.
"""
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = "data"

files = {
    "customer_info": "customer_info.csv",
    "usage_data": "usage_data.csv",
    "churn_labels": "churn_labels.csv",
}

frames = {}
for name, fn in files.items():
    df = pd.read_csv(f"{DATA}/{fn}")
    frames[name] = df
    print("=" * 78)
    print(f"{name}  ({fn})")
    print("=" * 78)
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"columns: {list(df.columns)}")
    print(f"duplicate full rows: {df.duplicated().sum():,}")
    print(f"duplicate CustomerID rows: {df['CustomerID'].duplicated().sum():,}")
    print(f"unique CustomerID: {df['CustomerID'].nunique():,}")
    print()

    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    print(prof.to_string())
    print()

    # raw value ranges for numeric-ish columns
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            print(f"  {c}: min={s.min()} max={s.max()} mean={s.mean():.3f} "
                  f"zeros={int((s == 0).sum()):,} negatives={int((s < 0).sum()):,}")
    print()

    # categorical value counts (small cardinality)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 20:
            print(f"  {c} value_counts (dropna=False):")
            print(df[c].value_counts(dropna=False).to_string())
            print()

print("=" * 78)
print("KEY / COVERAGE RELATIONSHIPS")
print("=" * 78)
ci, ud, cl = frames["customer_info"], frames["usage_data"], frames["churn_labels"]
si, su, sl = set(ci.CustomerID), set(ud.CustomerID), set(cl.CustomerID)

print(f"customer_info IDs : {len(si):,}")
print(f"usage_data   IDs  : {len(su):,}")
print(f"churn_labels IDs  : {len(sl):,}")
print()
print(f"info - usage  (in info, no usage rows) : {len(si - su):,}")
print(f"usage - info  (usage rows, no info)    : {len(su - si):,}")
print(f"info - labels (in info, no label)      : {len(si - sl):,}")
print(f"labels - info (labelled, no info row)  : {len(sl - si):,}")
print(f"usage - labels (usage, no label)       : {len(su - sl):,}")
print()
print(f"usage rows per customer: min={ud.groupby('CustomerID').size().min()}, "
      f"max={ud.groupby('CustomerID').size().max()}, "
      f"median={ud.groupby('CustomerID').size().median()}")
print()
print("Month value range:", ud.Month.min(), "->", ud.Month.max())
print("n unique months   :", ud.Month.nunique())
print("SignupDate range  :", ci.SignupDate.min(), "->", ci.SignupDate.max())
print()
print("Churn label distribution:")
print(cl.Churn.value_counts(dropna=False).to_string())
print(f"churn rate: {cl.Churn.mean():.4f}")
