#!/usr/bin/env python
"""SLAO stroke-CTA OSA feature analysis: exploratory, multidomain, MACE + etiology.

Mirrors the aortic SLAO analysis (aorta_cta_radiomics/scripts/build_slao_*),
but for the stroke_cta_osa upper-airway / tongue / neck-adiposity / skeletal
features. It reuses the clinical/MACE/etiology outcomes already joined for the
aorta pipeline (keyed by ``case_id = sub-XXXX``), so no REDCap re-touch.

Analyses (all written to --outdir):

  1. exploratory_associations.csv  — per-feature univariable logistic ORs (per
     SD), 95% CI, p, BH-FDR q, in-sample AUC, for each outcome (MACE + subtype).
  2. domain_burden_or.csv          — evidence-domain composite burden scores
     (airway / tongue / fat / skeletal / soft_tissue), univariable + adjusted
     multidomain ORs and AUC.
  3. etiology_predictor_models.csv — SLAO etiology *types as predictors* of
     MACE: logistic ORs + AUC, Cox HRs + survival C-index, and the incremental
     value of adding the OSA multidomain score.
  4. model_discrimination.csv      — AUC / C-index summary for the headline
     models.
  5. analysis_summary.md           — human-readable digest.

Research prototype — associations are exploratory and not adjusted for the full
confounder set; treat as hypothesis-generating.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
AORTA_OUT = REPO / "aorta_cta_radiomics" / "outputs" / "aorta_batch_run"
DEFAULT_FEATURES = REPO / "outputs" / "slao_cta_osa" / "features.csv"
# MACE endpoints come from the fuller MACE outcomes table (91 mace_primary /
# 406 cases); stroke-subtype (etiology) labels come from the etiology table.
DEFAULT_MACE = AORTA_OUT / "mace_slao" / "slao_mace_outcomes.csv"
DEFAULT_ETIOLOGY = AORTA_OUT / "etiology_slao" / "slao_etiology_aorta_modeling.csv"
DEFAULT_OUTDIR = REPO / "outputs" / "slao_cta_osa" / "analysis"

MACE_COLS = ["case_id", "mace_primary", "net_adverse_event", "all_cause_death",
             "mace_primary_time_days"]
ETI_COLS = ["case_id", "KAF", "AFDAS", "ECG_AF", "ESUS",
            "mace_plus_heart_failure", "stroke_mechanism"]
ETIOLOGY_INDICATORS = ["KAF", "AFDAS", "ECG_AF", "ESUS"]

# Minimum coverage / event counts to attempt a model.
MIN_COVERAGE = 0.50
MIN_EVENTS = 10

# Substrings marking scan-geometry / index / bookkeeping columns that are NOT
# anatomical features — they track FOV, slice position, or scan length and would
# inject field-of-view confounding into the associations. Also drop mm3 volumes
# (duplicates of the ml columns) and config echoes.
EXCLUDE_SUBSTR = (
    "slice_index", "_z_mm", "z_hi_index", "z_lo_index", "voxel_count",
    "n_slices", "profile", "threshold_used", "hu_min_used", "hu_max_used",
    "_volume_mm3", "_z_lo_index", "_z_hi_index", "csa_slice",
)


# --- feature domain (self-contained, by name prefix) -------------------------

def feature_domain(name: str) -> str:
    n = name.lower()
    if n.startswith(("airway_", "retropalatal_", "retroglossal_", "retrolingual_",
                     "nasopharyngeal_", "hypopharyngeal_")):
        return "airway"
    if n.startswith(("tongue_", "lingual_tonsil_")):
        return "tongue"
    if n.startswith("fat_"):
        return "fat"
    if n.startswith(("hyoid_", "mandible_", "mandibular_", "cervicomandibular_",
                     "neck_", "skeletal_", "laryngeal_", "hard_palate_",
                     "posterior_nasal_spine_")):
        return "skeletal"
    if n.startswith(("soft_palate_", "uvula_", "palatine_tonsil_",
                     "lateral_pharyngeal_", "lateral_wall_")):
        return "soft_tissue"
    return ""


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns with a domain, adequate coverage and variance."""
    out = []
    for c in df.columns:
        if not feature_domain(c):
            continue
        if any(sub in c.lower() for sub in EXCLUDE_SUBSTR):
            continue  # scan-geometry / index / bookkeeping, not anatomy
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < MIN_COVERAGE:
            continue
        if s.dropna().nunique() < 5 or s.std(skipna=True) in (0, np.nan):
            continue
        out.append(c)
    return out


def cv_auc(y: pd.Series, X: pd.DataFrame, n_splits: int = 5) -> float:
    """Cross-validated AUC (the binary C-statistic) — honest discrimination."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if d["y"].nunique() < 2 or int(d["y"].sum()) < MIN_EVENTS or len(d) < 40:
        return float("nan")
    yv = d["y"].astype(int).values
    Xv = d.drop(columns="y").values
    # standardise within-fold via pipeline-free z (predictors already ~z)
    skf = StratifiedKFold(n_splits=min(n_splits, int(yv.sum())), shuffle=True,
                          random_state=0)
    preds = np.full(len(yv), np.nan)
    for tr, te in skf.split(Xv, yv):
        clf = LogisticRegression(max_iter=200, C=1.0)
        try:
            clf.fit(Xv[tr], yv[tr])
            preds[te] = clf.predict_proba(Xv[te])[:, 1]
        except Exception:
            return float("nan")
    ok = ~np.isnan(preds)
    return float(roc_auc_score(yv[ok], preds[ok])) if ok.sum() > 10 else float("nan")


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    return (s - mu) / sd if sd and not np.isnan(sd) else s * np.nan


# --- modelling helpers -------------------------------------------------------

def logit_or(y: pd.Series, X: pd.DataFrame):
    """Fit logistic regression; return (params, conf_int, pvalues, auc) or None."""
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if d["y"].nunique() < 2 or int(d["y"].sum()) < MIN_EVENTS or len(d) < 30:
        return None
    Xd = sm.add_constant(d.drop(columns="y"), has_constant="add")
    try:
        res = sm.Logit(d["y"], Xd).fit(disp=0, maxiter=200)
        auc = roc_auc_score(d["y"], res.predict(Xd))
        return res, auc, len(d), int(d["y"].sum())
    except Exception:
        return None


def cox_hr(frame: pd.DataFrame, duration: str, event: str, cols: list[str]):
    """Fit Cox PH; return (fitted, concordance) or None."""
    from lifelines import CoxPHFitter
    d = frame[[duration, event] + cols].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[d[duration] > 0]
    if int(d[event].sum()) < MIN_EVENTS or len(d) < 30:
        return None
    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(d, duration_col=duration, event_col=event)
        return cph, float(cph.concordance_index_), len(d), int(d[event].sum())
    except Exception:
        return None


def bh_fdr(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values (monotone in p)."""
    p = pd.to_numeric(pvals, errors="coerce")
    ok = p.notna()
    pv = p[ok].sort_values()
    m = len(pv)
    if m == 0:
        return pd.Series(np.nan, index=p.index)
    ranks = np.arange(1, m + 1)
    q = pv.values * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0, 1)
    out = pd.Series(np.nan, index=p.index)
    out.loc[pv.index] = q
    return out


# --- build outcomes ----------------------------------------------------------

def build_outcomes(out_df: pd.DataFrame) -> pd.DataFrame:
    o = out_df.copy()
    for c in ETIOLOGY_INDICATORS + ["mace_primary", "net_adverse_event",
                                    "mace_plus_heart_failure", "all_cause_death"]:
        if c in o:
            o[c] = pd.to_numeric(o[c], errors="coerce")
    # binary derived subtype outcomes
    ind = o[ETIOLOGY_INDICATORS].fillna(0)
    o["etiology_af_related"] = (ind[["KAF", "AFDAS", "ECG_AF"]].sum(axis=1) > 0).astype(int)
    o["etiology_esus"] = (ind["ESUS"] > 0).astype(int)
    o["etiology_afdas_or_ecg_af"] = (ind[["AFDAS", "ECG_AF"]].sum(axis=1) > 0).astype(int)
    o["etiology_kaf"] = (ind["KAF"] > 0).astype(int)
    # single categorical etiology type (mutually exclusive)
    def _etype(r):
        for e in ETIOLOGY_INDICATORS:
            if r.get(e, 0) == 1:
                return e
        return "Other"
    o["etiology_type"] = ind.apply(_etype, axis=1)
    o["all_cause_death"] = o["all_cause_death"].fillna(0).clip(0, 1)
    return o


BINARY_OUTCOMES = {
    "mace_primary": "MACE (primary composite)",
    "mace_plus_heart_failure": "MACE + heart failure",
    "etiology_af_related": "AF-related etiology (vs ESUS/other)",
    "etiology_esus": "ESUS etiology",
    "etiology_afdas_or_ecg_af": "AFDAS or new ECG-AF (AFDAS spectrum)",
}


# --- analyses ----------------------------------------------------------------

def run_exploratory(df, feats, outcomes):
    rows = []
    for oc, oc_label in outcomes.items():
        if oc not in df:
            continue
        y = pd.to_numeric(df[oc], errors="coerce")
        recs = []
        for f in feats:
            fit = logit_or(y, zscore(df[f]).to_frame(f))
            if fit is None:
                continue
            res, auc, n, npos = fit
            if f not in res.params.index:
                continue
            beta = res.params[f]
            lo, hi = res.conf_int().loc[f]
            recs.append(dict(outcome=oc, outcome_label=oc_label, feature=f,
                             domain=feature_domain(f), n=n, n_events=npos,
                             or_per_sd=np.exp(beta), ci_low=np.exp(lo),
                             ci_high=np.exp(hi), p_value=res.pvalues[f], auc=auc))
        sub = pd.DataFrame(recs)
        if not sub.empty:
            sub["q_bh"] = bh_fdr(sub["p_value"])
            rows.append(sub)
    return (pd.concat(rows, ignore_index=True)
            .sort_values(["outcome", "p_value"]) if rows else pd.DataFrame())


def build_domain_scores(df, feats):
    """Per-domain composite = mean of z-scored features in that domain."""
    domains = {}
    for f in feats:
        domains.setdefault(feature_domain(f), []).append(f)
    scores = pd.DataFrame(index=df.index)
    for dom, cols in domains.items():
        z = pd.concat([zscore(df[c]) for c in cols], axis=1)
        # require at least one non-null feature in the domain for a case
        scores[f"{dom}_score"] = z.mean(axis=1, skipna=True)
    return scores, {d: cols for d, cols in domains.items()}


def run_domain_or(df, scores, outcomes):
    score_cols = list(scores.columns)
    frame = pd.concat([df, scores], axis=1)
    rows = []
    for oc, oc_label in outcomes.items():
        if oc not in frame:
            continue
        y = pd.to_numeric(frame[oc], errors="coerce")
        # univariable per domain
        for sc in score_cols:
            fit = logit_or(y, frame[[sc]])
            if fit is None:
                continue
            res, auc, n, npos = fit
            if sc not in res.params.index:
                continue
            lo, hi = res.conf_int().loc[sc]
            rows.append(dict(outcome=oc, outcome_label=oc_label, model="univariable",
                             term=sc, n=n, n_events=npos, or_per_sd=np.exp(res.params[sc]),
                             ci_low=np.exp(lo), ci_high=np.exp(hi),
                             p_value=res.pvalues[sc], model_auc=auc))
        # adjusted multidomain model (all domain scores together); report the
        # cross-validated AUC (honest C-statistic) rather than in-sample.
        fit = logit_or(y, frame[score_cols])
        if fit is not None:
            res, _insample, n, npos = fit
            auc = cv_auc(y, frame[score_cols])
            for sc in score_cols:
                if sc not in res.params.index:
                    continue
                lo, hi = res.conf_int().loc[sc]
                rows.append(dict(outcome=oc, outcome_label=oc_label,
                                 model="multidomain_adjusted", term=sc, n=n,
                                 n_events=npos, or_per_sd=np.exp(res.params[sc]),
                                 ci_low=np.exp(lo), ci_high=np.exp(hi),
                                 p_value=res.pvalues[sc], model_auc=auc))
    return pd.DataFrame(rows)


def run_etiology_predictors(df, scores):
    """SLAO etiology *types as predictors* of MACE.

    Reliable follow-up time for censored cases is not available in the SLAO
    outcome tables (``follow_up_duration_days`` is degenerate; the event time
    exists only for cases that had MACE), so a survival Cox / time C-index is not
    estimable. For these binary MACE endpoints the C-statistic equals the AUC, so
    we report the cross-validated AUC as the discrimination metric alongside the
    logistic odds ratios (ref = ESUS).
    """
    rows = []
    frame = pd.concat([df, scores], axis=1)
    et = pd.get_dummies(frame["etiology_type"], prefix="et")
    et_cols = [c for c in et.columns if c not in ("et_ESUS", "et_Other")]
    osa_score = zscore(scores.mean(axis=1)).rename("osa_multidomain_score")

    for oc in ["mace_primary", "net_adverse_event", "mace_plus_heart_failure"]:
        if oc not in frame:
            continue
        y = pd.to_numeric(frame[oc], errors="coerce")
        Xe = et[et_cols].astype(float)
        # etiology-only
        fit = logit_or(y, Xe)
        if fit is not None:
            res, _ins, n, npos = fit
            auc = cv_auc(y, Xe)
            for term in et_cols:
                if term not in res.params.index:
                    continue
                lo, hi = res.conf_int().loc[term]
                rows.append(dict(outcome=oc, model="etiology_only",
                                 term=term.replace("et_", "") + " vs ESUS",
                                 n=n, n_events=npos, or_=np.exp(res.params[term]),
                                 ci_low=np.exp(lo), ci_high=np.exp(hi),
                                 p_value=res.pvalues[term], cstat_cv_auc=auc))
        # etiology + OSA multidomain burden (incremental discrimination)
        Xc = pd.concat([Xe, osa_score], axis=1)
        fit = logit_or(y, Xc)
        if fit is not None:
            res, _ins, n, npos = fit
            auc = cv_auc(y, Xc)
            t = "osa_multidomain_score"
            if t in res.params.index:
                lo, hi = res.conf_int().loc[t]
                rows.append(dict(outcome=oc, model="etiology_plus_osa",
                                 term="OSA multidomain (per SD)", n=n, n_events=npos,
                                 or_=np.exp(res.params[t]), ci_low=np.exp(lo),
                                 ci_high=np.exp(hi), p_value=res.pvalues[t],
                                 cstat_cv_auc=auc))
        # OSA-only reference discrimination
        fit = logit_or(y, osa_score.to_frame())
        if fit is not None:
            res, _ins, n, npos = fit
            auc = cv_auc(y, osa_score.to_frame())
            t = "osa_multidomain_score"
            if t in res.params.index:
                lo, hi = res.conf_int().loc[t]
                rows.append(dict(outcome=oc, model="osa_only",
                                 term="OSA multidomain (per SD)", n=n, n_events=npos,
                                 or_=np.exp(res.params[t]), ci_low=np.exp(lo),
                                 ci_high=np.exp(hi), p_value=res.pvalues[t],
                                 cstat_cv_auc=auc))
    return pd.DataFrame(rows)


def discrimination_summary(domain_or, etiology, outcomes):
    """Cross-validated AUC (= binary C-statistic) for the headline models."""
    rows = []
    for oc, oc_label in outcomes.items():
        sub = domain_or[(domain_or.outcome == oc) &
                        (domain_or.model == "multidomain_adjusted")]
        if not sub.empty and pd.notna(sub["model_auc"].iloc[0]):
            rows.append(dict(outcome=oc, outcome_label=oc_label,
                             model="OSA multidomain", cv_auc=round(float(sub["model_auc"].iloc[0]), 3)))
    if not etiology.empty:
        for (oc, model), g in etiology.groupby(["outcome", "model"]):
            val = g["cstat_cv_auc"].dropna()
            if not val.empty:
                rows.append(dict(outcome=oc, outcome_label=oc,
                                 model=model, cv_auc=round(float(val.iloc[0]), 3)))
    return pd.DataFrame(rows).drop_duplicates()


def write_summary(path, meta, explor, domain_or, etiology, disc, domain_map):
    L = []
    L.append("# SLAO stroke-CTA OSA feature analysis\n")
    L.append(f"- Cases analysed (features ∩ outcomes): **{meta['n']}**")
    L.append(f"- OSA numeric features used: **{meta['n_feats']}** across domains: "
             + ", ".join(f"{d} ({len(c)})" for d, c in domain_map.items()) + "\n")
    L.append("> Exploratory / hypothesis-generating. ORs are per 1 SD. Not adjusted "
             "for the full clinical confounder set.\n")

    L.append("## Discrimination summary (AUC / C-index)\n")
    if not disc.empty:
        L.append(disc.to_markdown(index=False))
    L.append("")

    L.append("## Multidomain burden — adjusted ORs per outcome\n")
    for oc, oc_label in BINARY_OUTCOMES.items():
        sub = domain_or[(domain_or.outcome == oc) &
                        (domain_or.model == "multidomain_adjusted")]
        if sub.empty:
            continue
        L.append(f"### {oc_label} (n={int(sub['n'].iloc[0])}, "
                 f"events={int(sub['n_events'].iloc[0])}, "
                 f"AUC={sub['model_auc'].iloc[0]:.3f})")
        t = sub[["term", "or_per_sd", "ci_low", "ci_high", "p_value"]].copy()
        for c in ["or_per_sd", "ci_low", "ci_high"]:
            t[c] = t[c].round(2)
        t["p_value"] = t["p_value"].round(3)
        L.append(t.to_markdown(index=False))
        L.append("")

    L.append("## SLAO etiology types as predictors of MACE\n")
    L.append("_Odds ratios vs ESUS reference; discrimination = 5-fold CV AUC "
             "(the binary C-statistic). Survival Cox/time C-index not estimable — "
             "no follow-up time for censored cases in the SLAO tables._\n")
    if not etiology.empty:
        e = etiology.copy()
        for c in ["or_", "ci_low", "ci_high", "cstat_cv_auc"]:
            e[c] = pd.to_numeric(e[c], errors="coerce").round(3)
        e["p_value"] = pd.to_numeric(e["p_value"], errors="coerce").round(4)
        L.append(e[["outcome", "model", "term", "n", "n_events", "or_", "ci_low",
                    "ci_high", "p_value", "cstat_cv_auc"]].to_markdown(index=False))
    L.append("")

    L.append("## Top exploratory feature associations (q<0.10)\n")
    if not explor.empty and (explor["q_bh"] < 0.10).any():
        sig = explor[explor["q_bh"] < 0.10].sort_values(["outcome", "q_bh"]).copy()
        for c in ["or_per_sd", "ci_low", "ci_high", "auc"]:
            sig[c] = sig[c].round(2)
        sig["p_value"] = sig["p_value"].round(4); sig["q_bh"] = sig["q_bh"].round(3)
        L.append(sig[["outcome", "outcome_label", "feature", "domain", "or_per_sd",
                      "ci_low", "ci_high", "p_value", "q_bh", "auc"]]
                 .head(40).to_markdown(index=False))
    else:
        L.append("_No feature survived BH-FDR q<0.10 (expected at this n and with "
                 "tongue/mandible/skeletal mask-dependent features missing). See "
                 "exploratory_associations.csv for the full ranked list._")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--mace", type=Path, default=DEFAULT_MACE)
    ap.add_argument("--etiology", type=Path, default=DEFAULT_ETIOLOGY)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    feats_df = pd.read_csv(args.features)
    feats_df["case_id"] = feats_df["patient_id"].str.replace(
        "_acq-CTA_ct.nii.gz", "", regex=False)

    mace = pd.read_csv(args.mace)
    mace = mace[[c for c in MACE_COLS if c in mace.columns]]
    eti = pd.read_csv(args.etiology)
    eti = eti[[c for c in ETI_COLS if c in eti.columns]]
    out_df = mace.merge(eti, on="case_id", how="outer")
    out_df = build_outcomes(out_df)

    df = feats_df.merge(out_df, on="case_id", how="inner")
    feats = numeric_feature_columns(feats_df)
    scores, domain_map = build_domain_scores(df, feats)

    explor = run_exploratory(df, feats, BINARY_OUTCOMES)
    domain_or = run_domain_or(df, scores, BINARY_OUTCOMES)
    etiology = run_etiology_predictors(df, scores)
    disc = discrimination_summary(domain_or, etiology, BINARY_OUTCOMES)

    explor.to_csv(args.outdir / "exploratory_associations.csv", index=False)
    domain_or.to_csv(args.outdir / "domain_burden_or.csv", index=False)
    etiology.to_csv(args.outdir / "etiology_predictor_models.csv", index=False)
    disc.to_csv(args.outdir / "model_discrimination.csv", index=False)
    write_summary(args.outdir / "analysis_summary.md",
                  {"n": len(df), "n_feats": len(feats)},
                  explor, domain_or, etiology, disc, domain_map)

    print(f"n_cases={len(df)}  n_features={len(feats)}  "
          f"domains={ {d: len(c) for d, c in domain_map.items()} }")
    print("wrote:", *(p.name for p in sorted(args.outdir.glob("*"))), sep="\n  ")


if __name__ == "__main__":
    main()
