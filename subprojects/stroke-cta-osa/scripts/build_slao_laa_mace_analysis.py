#!/usr/bin/env python
"""SLAO LAA (SLAAO) features/types vs MACE.

Joins Luciano Sposato's final SLAAO dataset (LAA types + continuous LAA geometry)
to the 91-event MACE outcomes table. The join key is validated:
``Study_ID == record_id`` (sex matches 381/381 after correcting the inverse sex
coding; age_at_event r=0.999). ``sex_num`` in the SLAAO file is coded
**inversely** (1=Female, 0=Male) relative to the MACE/REDCap table — corrected
here before any adjustment.

Outputs (--outdir):
  1. laa_types_vs_mace.csv        — SLAAO types → MACE, crude and age/sex-adjusted
                                     OR + AUC.
  2. laa_continuous_vs_mace.csv   — continuous LAA features → MACE, OR per SD,
                                     95% CI, p, BH-FDR q, crude + adjusted AUC.
  3. laa_mace_summary.md          — digest.

Exploratory / hypothesis-generating; univariable + minimally (age, sex) adjusted.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SLAAO = Path.home() / "Desktop" / "SLAO analysis" / "SLAAO_Final_Dataset.csv"
DEFAULT_MACE = (REPO / "aorta_cta_radiomics" / "outputs" / "aorta_batch_run"
                / "mace_slao" / "slao_mace_outcomes.csv")
DEFAULT_OUTDIR = REPO / "outputs" / "slao_cta_osa" / "analysis" / "laa"

MIN_COVERAGE = 0.40
MIN_EVENTS = 10

SLAAO_TYPES = ["SLAAO_overall", "SLAAO_Type1", "SLAAO_Type2",
               "SLAAO_Type2a", "SLAAO_Type2b", "SLAAO_Type2c"]

# Base continuous LAA metrics; each may appear per scan (CCT/CTT/eCTA) and/or
# unsuffixed. We coalesce across scans (mean of available) per patient.
LAA_BASE_METRICS = [
    "laa_vol_ml", "laa_ao_ratio", "laa_pa_ratio", "hu_laa",
    "laa_elongation", "laa_flatness", "laa_bend_angle",
    "laa_la_bend_angle", "ostium_LAA_bend_angle",
]


def coalesce_metric(df: pd.DataFrame, base: str) -> pd.Series:
    """Mean across all columns whose name is `base` or `base_<scan>`."""
    cols = [c for c in df.columns
            if c == base or c.startswith(base + "_")]
    # keep only numeric-looking scan variants (exclude e.g. *_num flags)
    cols = [c for c in cols if not c.endswith("_num")]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.mean(axis=1, skipna=True)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    return (s - s.mean(skipna=True)) / sd if sd and not np.isnan(sd) else s * np.nan


def logit(y: pd.Series, X: pd.DataFrame):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if d["y"].nunique() < 2 or int(d["y"].sum()) < MIN_EVENTS or len(d) < 40:
        return None
    Xd = sm.add_constant(d.drop(columns="y"), has_constant="add")
    try:
        res = sm.Logit(d["y"], Xd).fit(disp=0, maxiter=200)
        auc = roc_auc_score(d["y"], res.predict(Xd))
        return res, auc, len(d), int(d["y"].sum())
    except Exception:
        return None


def bh_fdr(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    ok = p.notna(); pv = p[ok].sort_values(); m = len(pv)
    if m == 0:
        return pd.Series(np.nan, index=p.index)
    q = pv.values * m / np.arange(1, m + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    out = pd.Series(np.nan, index=p.index); out.loc[pv.index] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slaao", type=Path, default=DEFAULT_SLAAO)
    ap.add_argument("--mace", type=Path, default=DEFAULT_MACE)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    slaa = pd.read_csv(args.slaao, dtype=str, low_memory=False)
    slaa["record_id"] = slaa["Study_ID"].astype(str)
    mace = pd.read_csv(args.mace, dtype=str)
    mace["record_id"] = mace["record_id"].astype(str)
    mace["mace"] = pd.to_numeric(mace["mace_primary"], errors="coerce")
    # corrected sex: REDCap/MACE M/F -> 1/0 (male=1)
    sex_m = mace["sex"].str.upper().map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0,
                                         "1": 1, "0": 0})
    out = pd.DataFrame({"record_id": mace["record_id"], "mace": mace["mace"],
                        "cov_age": pd.to_numeric(mace["age"], errors="coerce"),
                        "cov_sex": sex_m})

    df = slaa.merge(out, on="record_id", how="inner")
    df = df.dropna(subset=["mace"])
    n, nev = len(df), int(df["mace"].sum())

    # ---- 1. SLAAO types -> MACE (crude + age/sex-adjusted) ----
    trows = []
    for t in SLAAO_TYPES:
        x = pd.to_numeric(df[t], errors="coerce")
        if x.notna().sum() == 0 or x.dropna().nunique() < 2:
            continue
        npos = int((x == 1).sum())
        crude = logit(df["mace"], x.rename(t).to_frame())
        adj = logit(df["mace"], pd.concat(
            [x.rename(t), df["cov_age"], df["cov_sex"]], axis=1))
        row = dict(predictor=t, n_positive=npos, n=n, n_events=nev)
        if crude:
            r, auc, *_ = crude
            lo, hi = r.conf_int().loc[t]
            row.update(or_crude=np.exp(r.params[t]), ci_low_crude=np.exp(lo),
                       ci_high_crude=np.exp(hi), p_crude=r.pvalues[t], auc_crude=auc)
        if adj and t in adj[0].params.index:
            r, auc, *_ = adj
            lo, hi = r.conf_int().loc[t]
            row.update(or_adj=np.exp(r.params[t]), ci_low_adj=np.exp(lo),
                       ci_high_adj=np.exp(hi), p_adj=r.pvalues[t], auc_adj_model=auc)
        trows.append(row)
    types_tbl = pd.DataFrame(trows)

    # ---- 2. Continuous LAA features -> MACE ----
    crows = []
    for base in LAA_BASE_METRICS:
        val = coalesce_metric(df, base)
        if val.notna().mean() < MIN_COVERAGE or val.dropna().nunique() < 5:
            continue
        z = zscore(val)
        crude = logit(df["mace"], z.rename(base).to_frame())
        adj = logit(df["mace"], pd.concat([z.rename(base), df["cov_age"], df["cov_sex"]], axis=1))
        row = dict(feature=base, n_nonnull=int(val.notna().sum()))
        if crude:
            r, auc, nn, ne = crude
            lo, hi = r.conf_int().loc[base]
            row.update(n=nn, n_events=ne, or_per_sd=np.exp(r.params[base]),
                       ci_low=np.exp(lo), ci_high=np.exp(hi),
                       p_value=r.pvalues[base], auc_crude=auc)
        if adj and base in adj[0].params.index:
            r, auc, *_ = adj
            lo, hi = r.conf_int().loc[base]
            row.update(or_adj_per_sd=np.exp(r.params[base]), ci_low_adj=np.exp(lo),
                       ci_high_adj=np.exp(hi), p_adj=r.pvalues[base], auc_adj_model=auc)
        crows.append(row)
    cont_tbl = pd.DataFrame(crows)
    if not cont_tbl.empty and "p_value" in cont_tbl:
        cont_tbl["q_bh"] = bh_fdr(cont_tbl["p_value"])
        cont_tbl = cont_tbl.sort_values("p_value")

    types_tbl.to_csv(args.outdir / "laa_types_vs_mace.csv", index=False)
    cont_tbl.to_csv(args.outdir / "laa_continuous_vs_mace.csv", index=False)

    # ---- summary ----
    L = [f"# SLAO LAA (SLAAO) features vs MACE\n",
         f"- Matched cases (SLAAO ∩ MACE): **{n}**, MACE events: **{nev}**",
         "- Join `Study_ID == record_id` validated (sex 381/381 after inverse-sex "
         "correction; age_at_event r=0.999). Sex corrected (male=1).\n",
         "> Exploratory. Continuous ORs per 1 SD; adjusted = + age + sex.\n",
         "## SLAAO types → MACE\n"]
    if not types_tbl.empty:
        t = types_tbl.copy()
        for c in t.columns:
            if c.startswith(("or", "ci", "auc")):
                t[c] = pd.to_numeric(t[c], errors="coerce").round(2)
            if c.startswith("p_"):
                t[c] = pd.to_numeric(t[c], errors="coerce").round(3)
        cols = [c for c in ["predictor", "n_positive", "or_crude", "ci_low_crude",
                            "ci_high_crude", "p_crude", "auc_crude", "or_adj",
                            "p_adj", "auc_adj_model"] if c in t.columns]
        L.append(t[cols].to_markdown(index=False))
    L.append("\n## Continuous LAA features → MACE (ranked)\n")
    if not cont_tbl.empty:
        t = cont_tbl.copy()
        for c in ["or_per_sd", "ci_low", "ci_high", "auc_crude", "or_adj_per_sd",
                  "auc_adj_model"]:
            if c in t:
                t[c] = pd.to_numeric(t[c], errors="coerce").round(2)
        for c in ["p_value", "p_adj", "q_bh"]:
            if c in t:
                t[c] = pd.to_numeric(t[c], errors="coerce").round(3)
        cols = [c for c in ["feature", "n", "n_events", "or_per_sd", "ci_low",
                            "ci_high", "p_value", "q_bh", "auc_crude",
                            "or_adj_per_sd", "p_adj"] if c in t.columns]
        L.append(t[cols].to_markdown(index=False))
    (args.outdir / "laa_mace_summary.md").write_text("\n".join(L) + "\n")

    print(f"matched n={n} MACE={nev}")
    print("types tested:", len(types_tbl), "| continuous features tested:", len(cont_tbl))
    print("wrote:", *(p.name for p in sorted(args.outdir.glob('*'))), sep="\n  ")


if __name__ == "__main__":
    main()
