"""One-off patch: add ratio_la_aorta_entropy/mean/median to existing radiomics CSV."""
import pandas as pd
from pathlib import Path

CSV = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS\derivatives\radiomics_ts_laa.csv")

df = pd.read_csv(CSV)

for feat_key, feat_slug in [("Entropy", "entropy"), ("Mean", "mean"), ("Median", "median")]:
    la_col  = f"la_original_firstorder_{feat_key}"
    ao_col  = f"aorta_original_firstorder_{feat_key}"
    out_col = f"ratio_la_aorta_{feat_slug}"
    df[out_col] = pd.to_numeric(df[la_col], errors="coerce") / pd.to_numeric(df[ao_col], errors="coerce")
    n_ok  = df[out_col].notna().sum()
    n_nan = df[out_col].isna().sum()
    print(f"{out_col}: {n_ok} computed, {n_nan} NaN")

df.to_csv(CSV, index=False)
print(f"\nSaved: {CSV}  ({len(df)} rows)")
