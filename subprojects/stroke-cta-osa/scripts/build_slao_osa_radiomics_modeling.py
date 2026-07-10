#!/usr/bin/env python
"""SLAO OSA radiomics-style modeling, mirroring the aorta MACE + etiology pipeline.

Outcome/etiology/covariate labels are inherited from the aorta modeling tables
(joined on ``case_id``) so the cohort and outcome definitions are byte-for-byte the
same as the aorta analysis. On the OSA imaging side we run the identical recipe:

  per training fold -> drop missingness>0.25 / near-zero variance
                    -> rank by |standardized mean difference|
                    -> greedy spearman<0.85 collinearity prune (min pair n=40)
                    -> keep top-k -> elastic-net logistic (C=0.2, l1_ratio=0.5)
                    -> out-of-fold probability;
  plus a train-standardized signed-z score, cross-fitted Platt recalibration,
  a pooled all-imaging model (top-k=24), a domain-sum score, 100x stability
  selection, and age/sex-adjusted burden odds ratios.

Domains (OSA analogue of the aorta calcium/fat/wall domains):
  airway | tongue | fat | skeletal

RESEARCH PROTOTYPE - NOT FOR CLINICAL DIAGNOSIS.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = REPO / "outputs/slao_cta_osa/features_all_exploratory.csv"
DEFAULT_MACE = REPO / "aorta_cta_radiomics/outputs/aorta_batch_run/mace_slao/slao_mace_aorta_modeling.csv"
DEFAULT_ETI = REPO / "aorta_cta_radiomics/outputs/aorta_batch_run/etiology_slao/slao_etiology_aorta_modeling.csv"
DEFAULT_OUTDIR = REPO / "outputs/slao_cta_osa/modeling"

DOMAIN_ORDER = ["airway", "tongue", "fat", "skeletal"]
DOMAIN_LABELS = {
    "airway": "Upper airway",
    "tongue": "Tongue / soft tissue",
    "fat": "Peripharyngeal / cervical fat",
    "skeletal": "Craniofacial skeleton",
    "all_imaging": "All imaging",
    "domain_sum": "Domain sum",
}

# hyperparameters (identical to the aorta radiomics-score defaults)
TOP_K = 8
ALL_TOP_K = 24
FOLDS = 5
MAX_MISSING = 0.25
MIN_VAR = 1e-12
CORR_THRESH = 0.85
MIN_CORR_PAIR_N = 40
EN_C = 0.2
EN_L1 = 0.5
STAB_RESAMPLES = 100
STAB_FRACTION = 0.75
STAB_THRESHOLD = 0.50
SEED = 42


# --------------------------------------------------------------------------- #
# feature / domain identification
# --------------------------------------------------------------------------- #
def is_excluded(col: str) -> bool:
    l = col.lower()
    if l.startswith("qc_"):
        return True
    if l in {"pipeline", "pipeline_version", "config_hash", "processing_timestamp",
             "patient_id", "study_id", "scan_id", "input_path_hash", "input_kind",
             "airway_source", "airway_provider_notes", "case_id"}:
        return True
    if l.endswith(("_method", "_confidence", "_available", "_flag", "_notes",
                   "_hash", "_id", "_timestamp", "_version")):
        return True
    if "spacing" in l or "z_extent" in l or "_region" == l[-7:]:
        return True
    return False


def domain_of(col: str) -> str | None:
    l = col.lower()
    if any(k in l for k in ("mandible", "skeletal", "hyoid", "cervicomandibular", "enclosure")):
        return "skeletal"
    if l.startswith("fat_") or "_fat_" in l or "fat_fraction" in l or "adipose" in l:
        return "fat"
    if l.startswith("tongue") or "oral_cavity" in l or "genioglossus" in l:
        return "tongue"
    if l.startswith("airway") or any(k in l for k in ("retropalatal", "retroglossal", "retrolingual", "_csa")):
        return "airway"
    return None


def load_feature_domains(features_csv: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_csv(features_csv)
    df["case_id"] = df["patient_id"].astype(str).str.split("_").str[0]
    domains: dict[str, list[str]] = {d: [] for d in DOMAIN_ORDER}
    for col in df.columns:
        if is_excluded(col):
            continue
        dom = domain_of(col)
        if dom is None:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        n = ser.notna().sum()
        if n < MIN_CORR_PAIR_N or ser.nunique(dropna=True) <= 2:
            continue
        df[col] = ser
        domains[dom].append(col)
    domains = {d: cols for d, cols in domains.items() if cols}
    keep = ["case_id"] + [c for cols in domains.values() for c in cols]
    return df[keep].copy(), domains


# --------------------------------------------------------------------------- #
# modeling primitives
# --------------------------------------------------------------------------- #
def smd(x: np.ndarray, y: np.ndarray) -> float:
    """Standardized mean difference of feature x between y==1 and y==0."""
    a, b = x[y == 1], x[y == 0]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return 0.0
    sd = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2.0)
    if sd < 1e-9:
        return 0.0
    return float((np.nanmean(a) - np.nanmean(b)) / sd)


def screen_and_select(X: pd.DataFrame, y: np.ndarray, candidates: list[str], top_k: int) -> list[str]:
    """Missingness/variance screen -> |SMD| rank -> spearman collinearity prune -> top-k."""
    kept_pre = []
    for c in candidates:
        col = X[c].to_numpy(dtype=float)
        if np.mean(np.isnan(col)) > MAX_MISSING:
            continue
        if np.nanvar(col) <= MIN_VAR:
            continue
        kept_pre.append(c)
    ranked = sorted(kept_pre, key=lambda c: abs(smd(X[c].to_numpy(float), y)), reverse=True)
    selected: list[str] = []
    for cand in ranked:
        cvec = X[cand].to_numpy(float)
        drop = False
        for s in selected:
            svec = X[s].to_numpy(float)
            mask = ~np.isnan(cvec) & ~np.isnan(svec)
            if mask.sum() < MIN_CORR_PAIR_N:
                continue
            rho, _ = stats.spearmanr(cvec[mask], svec[mask])
            if np.isfinite(rho) and abs(rho) >= CORR_THRESH:
                drop = True
                break
        if not drop:
            selected.append(cand)
        if len(selected) >= top_k:
            break
    return selected


def _prep(train_X: pd.DataFrame, test_X: pd.DataFrame, cols: list[str]):
    med = train_X[cols].median(numeric_only=True)
    tr = train_X[cols].fillna(med)
    te = test_X[cols].fillna(med)
    mu, sd = tr.mean(), tr.std(ddof=0).replace(0, 1.0)
    return (tr - mu) / sd, (te - mu) / sd


def elastic_net():
    return LogisticRegression(
        penalty="elasticnet", solver="saga", C=EN_C, l1_ratio=EN_L1,
        max_iter=8000, tol=1e-3, random_state=SEED,
    )


def cross_fit_scores(X: pd.DataFrame, y: np.ndarray, domains: dict[str, list[str]],
                     seed: int = SEED) -> tuple[pd.DataFrame, list[str], dict]:
    """Return per-case OOF scores for each domain + all_imaging + domain_sum.

    Two score flavors per model: elastic-net OOF probability and signed-z sum.
    """
    n = len(y)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    all_cols = [c for cols in domains.values() for c in cols]
    models = {d: (cols, TOP_K) for d, cols in domains.items()}
    models["all_imaging"] = (all_cols, ALL_TOP_K)

    prob = {m: np.full(n, np.nan) for m in models}
    signed = {m: np.full(n, np.nan) for m in models}
    selected_counts: dict[str, dict[str, int]] = {m: {} for m in models}

    idx = np.arange(n)
    for tr_i, te_i in skf.split(idx, y):
        trX, teX = X.iloc[tr_i], X.iloc[te_i]
        ytr = y[tr_i]
        for m, (cols, k) in models.items():
            sel = screen_and_select(trX, ytr, cols, k)
            if not sel:
                continue
            for s in sel:
                selected_counts[m][s] = selected_counts[m].get(s, 0) + 1
            Ztr, Zte = _prep(trX, teX, sel)
            # elastic-net probability
            try:
                clf = elastic_net().fit(Ztr.values, ytr)
                prob[m][te_i] = clf.predict_proba(Zte.values)[:, 1]
            except Exception:
                pass
            # signed-z sum (orient each feature by train SMD)
            signs = np.array([np.sign(smd(trX[s].to_numpy(float), ytr)) or 1.0 for s in sel])
            zt = np.clip(Zte.values * signs, -8, 8)
            signed[m][te_i] = zt.mean(axis=1)

    score_df = pd.DataFrame({"case_id": X.index if X.index.name == "case_id" else np.arange(n)})
    score_df = pd.DataFrame(index=X.index)
    for m in models:
        score_df[f"{m}__elastic_net_probability_cv"] = prob[m]
        score_df[f"{m}__signed_z_cv"] = signed[m]
    # domain_sum = mean of the four domain signed-z (z-normalized first)
    dz = []
    for d in DOMAIN_ORDER:
        col = f"{d}__signed_z_cv"
        if col in score_df:
            v = score_df[col]
            dz.append((v - v.mean()) / (v.std(ddof=0) or 1.0))
    if dz:
        score_df["domain_sum__signed_z_cv"] = pd.concat(dz, axis=1).mean(axis=1)
    # platt recalibration (cross-fitted) for each probability score
    for m in list(models):
        raw = score_df[f"{m}__elastic_net_probability_cv"]
        score_df[f"{m}__elastic_net_probability_cv__platt_cv"] = platt_cv(raw.to_numpy(), y, seed)

    return score_df, all_cols, selected_counts


def platt_cv(raw: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    out = np.full(len(y), np.nan)
    ok = ~np.isnan(raw)
    if ok.sum() < 20:
        return out
    eps = 1e-6
    logit = np.log(np.clip(raw, eps, 1 - eps) / (1 - np.clip(raw, eps, 1 - eps)))
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed + 1)
    idx = np.where(ok)[0]
    for tr_rel, te_rel in skf.split(idx, y[idx]):
        tr, te = idx[tr_rel], idx[te_rel]
        try:
            lr = LogisticRegression(max_iter=1000).fit(logit[tr].reshape(-1, 1), y[tr])
            out[te] = lr.predict_proba(logit[te].reshape(-1, 1))[:, 1]
        except Exception:
            pass
    return out


def performance_rows(score_df: pd.DataFrame, y: np.ndarray, events_label: str) -> list[dict]:
    rows = []
    for col in score_df.columns:
        s = score_df[col].to_numpy(float)
        ok = ~np.isnan(s)
        if ok.sum() < 40 or len(np.unique(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], s[ok])
        rows.append({
            "score_name": col,
            "domain": col.split("__")[0],
            "n": int(ok.sum()),
            "events": int(y[ok].sum()),
            "auc": round(max(auc, 1 - auc), 6),
            "directional_auc": round(auc, 6),
            "average_precision": round(average_precision_score(y[ok], s[ok]), 6),
        })
    rows.sort(key=lambda r: r["auc"], reverse=True)
    return rows


def stability_selection(X: pd.DataFrame, y: np.ndarray, cols: list[str], top_k: int,
                        resamples: int = STAB_RESAMPLES) -> list[dict]:
    rng = np.random.default_rng(SEED)
    counts: dict[str, int] = {}
    n = len(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(resamples):
        tr = np.concatenate([
            rng.choice(pos, int(len(pos) * STAB_FRACTION), replace=False),
            rng.choice(neg, int(len(neg) * STAB_FRACTION), replace=False),
        ])
        sel = screen_and_select(X.iloc[tr], y[tr], cols, top_k)
        Ztr, _ = _prep(X.iloc[tr], X.iloc[tr], sel) if sel else (None, None)
        if not sel:
            continue
        try:
            clf = elastic_net().fit(Ztr.values, y[tr])
            nz = [s for s, c in zip(sel, clf.coef_[0]) if abs(c) > 1e-8]
        except Exception:
            nz = sel
        for s in nz:
            counts[s] = counts.get(s, 0) + 1
    out = [{"feature": k, "selection_probability": round(v / resamples, 3),
            "domain": domain_of(k)} for k, v in counts.items()]
    out.sort(key=lambda r: r["selection_probability"], reverse=True)
    return out


def burden_or(score: np.ndarray, y: np.ndarray, age: np.ndarray, sex: np.ndarray) -> list[dict]:
    """Odds ratio for high burden (>=median, >=top-quartile), unadjusted and age+sex adjusted."""
    import statsmodels.api as sm
    rows = []
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, yy = score[ok], y[ok]
    ag, sx = age[ok], sex[ok]
    for label, thr in [("above_median", np.nanmedian(s)), ("top_quartile", np.nanquantile(s, 0.75))]:
        high = (s >= thr).astype(float)
        for adj, X in [("none", high.reshape(-1, 1)),
                       ("age+sex", np.column_stack([high, ag, sx]))]:
            m = ~np.isnan(X).any(axis=1)
            try:
                Xc = sm.add_constant(X[m], has_constant="add")
                res = sm.Logit(yy[m], Xc).fit(disp=0)
                beta, se = res.params[1], res.bse[1]
                rows.append({
                    "contrast": f"high_burden_{label}", "adjustment": adj,
                    "n": int(m.sum()), "events": int(yy[m].sum()),
                    "or": round(float(np.exp(beta)), 4),
                    "ci_low": round(float(np.exp(beta - 1.96 * se)), 4),
                    "ci_high": round(float(np.exp(beta + 1.96 * se)), 4),
                    "p_value": round(float(res.pvalues[1]), 6),
                })
            except Exception:
                pass
    return rows


# --------------------------------------------------------------------------- #
# outcome loaders (inherit from aorta modeling tables)
# --------------------------------------------------------------------------- #
def load_outcomes(mace_csv: Path, eti_csv: Path):
    mace = pd.read_csv(mace_csv, low_memory=False)
    eti = pd.read_csv(eti_csv, low_memory=False)
    mace_keep = mace[["case_id", "age", "sex", "mace_primary", "mace_plus_heart_failure",
                      "net_adverse_event"]].copy()
    eti_cols = ["case_id", "age", "sex", "source_etiology_label", "etiology_cardioembolic_vs_esus"]
    eti_keep = eti[[c for c in eti_cols if c in eti.columns]].copy()
    return mace_keep, eti_keep


def coerce_sex(s: pd.Series) -> np.ndarray:
    def f(v):
        t = str(v).strip().lower()
        if t in ("1", "m", "male", "true"):
            return 1.0
        if t in ("0", "2", "f", "female", "false"):
            return 0.0
        try:
            return float(v)
        except Exception:
            return np.nan
    return s.map(f).to_numpy(float)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_mace(feat: pd.DataFrame, domains, mace: pd.DataFrame, outdir: Path):
    df = feat.merge(mace, on="case_id", how="inner")
    df = df.dropna(subset=["mace_primary"])
    y = df["mace_primary"].astype(float).to_numpy()
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    sex = coerce_sex(df["sex"])
    X = df.set_index("case_id")[[c for cols in domains.values() for c in cols]]

    scores, all_cols, sel_counts = cross_fit_scores(X, y, domains)
    perf = performance_rows(scores, y, "MACE")
    stab = stability_selection(X.reset_index(drop=True), y, all_cols, ALL_TOP_K)

    scores_out = scores.reset_index()
    scores_out.insert(1, "mace_primary", y.astype(int))
    (outdir / "mace").mkdir(parents=True, exist_ok=True)
    scores_out.to_csv(outdir / "mace/osa_mace_radiomics_scores.csv", index=False)
    pd.DataFrame(perf).to_csv(outdir / "mace/osa_mace_score_performance.csv", index=False)
    pd.DataFrame(stab).to_csv(outdir / "mace/osa_mace_stability_selection.csv", index=False)

    # burden ORs for each domain + pooled + domain_sum signed-z
    bor = []
    for col in scores.columns:
        if col.endswith("signed_z_cv") or col.endswith("elastic_net_probability_cv"):
            for r in burden_or(scores[col].to_numpy(float), y, age, sex):
                r = {"score_name": col, "domain": col.split("__")[0], **r}
                bor.append(r)
    pd.DataFrame(bor).to_csv(outdir / "mace/osa_mace_burden_or.csv", index=False)

    summary = {
        "analysis": "OSA MACE (mirrors aorta pipeline)",
        "features_csv": str(DEFAULT_FEATURES),
        "outcome": "mace_primary (inherited from aorta MACE modeling table)",
        "merged_rows": int(len(df)),
        "events": int(y.sum()),
        "nonevents": int((y == 0).sum()),
        "domains": {d: len(c) for d, c in domains.items()},
        "n_features": int(sum(len(c) for c in domains.values())),
        "hyperparameters": {"top_k": TOP_K, "all_top_k": ALL_TOP_K, "folds": FOLDS,
                            "elastic_net_c": EN_C, "l1_ratio": EN_L1,
                            "correlation_threshold": CORR_THRESH,
                            "stability_resamples": STAB_RESAMPLES},
        "top_scores": perf[:8],
        "top_stable_features": stab[:12],
    }
    (outdir / "mace/osa_mace_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_etiology(feat: pd.DataFrame, domains, eti: pd.DataFrame, outdir: Path):
    df = feat.merge(eti, on="case_id", how="inner")
    df = df[df["source_etiology_label"].notna()].copy()
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    sex = coerce_sex(df["sex"])
    Xfull = df.set_index("case_id")[[c for cols in domains.values() for c in cols]]
    label = df["source_etiology_label"].astype(str).str.strip()

    targets = {
        "cardioembolic_vs_esus": ("etiology_cardioembolic_vs_esus", None),  # special: from column
        "kaf": (None, {"KAF"}),
        "afdas": (None, {"AFDAS"}),
        "ecg_af": (None, {"New_ECG_AF", "ECG_AF"}),
        "esus": (None, {"ESUS"}),
        "afdas_or_ecg_af": (None, {"AFDAS", "New_ECG_AF", "ECG_AF"}),
    }
    (outdir / "etiology").mkdir(parents=True, exist_ok=True)
    all_perf, all_scores, summaries, all_bor, all_stab = [], {"case_id": Xfull.index}, {}, [], []
    all_cols = [c for cols in domains.values() for c in cols]

    for slug, (col, pos_labels) in targets.items():
        if col is not None:
            yv = df[col].astype(str).str.strip().str.lower()
            mask = yv.isin(["cardioembolic", "esus", "1", "0", "1.0", "0.0"])
            y = yv.map(lambda t: 1.0 if t in ("cardioembolic", "1", "1.0") else
                       (0.0 if t in ("esus", "0", "0.0") else np.nan)).to_numpy(float)
        else:
            y = label.isin(pos_labels).astype(float).to_numpy()
            mask = pd.Series(True, index=df.index)
        keep = ~np.isnan(y)
        Xt = Xfull.iloc[keep]
        yt = y[keep]
        if yt.sum() < 10 or (yt == 0).sum() < 10:
            continue
        scores, _, _ = cross_fit_scores(Xt, yt, domains, seed=SEED)
        perf = performance_rows(scores, yt, slug)
        for r in perf:
            r["target"] = slug
        all_perf.extend(perf)
        # store best score per target
        for col2 in scores.columns:
            all_scores.setdefault(f"{slug}__{col2}", pd.Series(np.nan, index=Xfull.index))
            all_scores[f"{slug}__{col2}"].loc[Xt.index] = scores[col2].values
        stab = stability_selection(Xt.reset_index(drop=True), yt, all_cols, ALL_TOP_K, resamples=60)
        for r in stab:
            r["target"] = slug
        all_stab.extend(stab)
        for col2 in scores.columns:
            if col2.endswith("signed_z_cv") or col2.endswith("elastic_net_probability_cv"):
                agv, sxv = age[keep], sex[keep]
                for r in burden_or(scores[col2].to_numpy(float), yt, agv, sxv):
                    all_bor.append({"target": slug, "score_name": col2,
                                    "domain": col2.split("__")[0], **r})
        summaries[slug] = {"positive": int(yt.sum()), "negative": int((yt == 0).sum()),
                           "best_auc": perf[0]["auc"] if perf else None,
                           "best_score": perf[0]["score_name"] if perf else None}

    pd.DataFrame(all_perf).to_csv(outdir / "etiology/osa_etiology_one_vs_rest_performance.csv", index=False)
    pd.DataFrame(all_bor).to_csv(outdir / "etiology/osa_etiology_burden_or.csv", index=False)
    pd.DataFrame(all_stab).to_csv(outdir / "etiology/osa_etiology_stability_selection.csv", index=False)
    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(outdir / "etiology/osa_etiology_scores.csv", index=False)
    summary = {
        "analysis": "OSA etiology (mirrors aorta pipeline)",
        "outcome": "one-vs-rest per source_etiology_label + cardioembolic_vs_esus",
        "modeled_rows": int(len(df)),
        "source_etiology_label_counts": label.value_counts().to_dict(),
        "domains": {d: len(c) for d, c in domains.items()},
        "targets": summaries,
    }
    (outdir / "etiology/osa_etiology_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--mace", type=Path, default=DEFAULT_MACE)
    ap.add_argument("--etiology", type=Path, default=DEFAULT_ETI)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    feat, domains = load_feature_domains(args.features)
    feat.to_csv(args.outdir / "osa_modeling_wide.csv", index=False)
    print(f"[osa-model] features: {sum(len(c) for c in domains.values())} across "
          f"{ {d: len(c) for d, c in domains.items()} }")

    mace_out, eti_out = load_outcomes(args.mace, args.etiology)
    print("[osa-model] running MACE ...")
    ms = run_mace(feat, domains, mace_out, args.outdir)
    print(f"[osa-model]   MACE merged={ms['merged_rows']} events={ms['events']} "
          f"best={ms['top_scores'][0]['score_name']} AUC={ms['top_scores'][0]['auc']}")
    print("[osa-model] running etiology ...")
    es = run_etiology(feat, domains, eti_out, args.outdir)
    for slug, s in es["targets"].items():
        print(f"[osa-model]   {slug:22s} pos={s['positive']:3d} AUC={s['best_auc']}")
    print("[osa-model] done ->", args.outdir)


if __name__ == "__main__":
    main()
