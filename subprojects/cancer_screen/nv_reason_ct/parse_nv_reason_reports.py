#!/usr/bin/env python3
"""Turn NV-Reason-CT JSON outputs into coarse screening flags that join the cancer_screen features table.

  python parse_nv_reason_reports.py --in-dir <dir with *_nvreason.json> --out <csv>

Flags are keyword-based (a reader still reads the report): nodule/mass mentioned, lymphadenopathy mentioned,
thyroid nodule mentioned, malignancy answer yes/no, largest size in mm quoted in each answer.
"""
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

NEG = re.compile(r"\b(no|without|absence of|not (?:seen|identified|present)|unremarkable|normal)\b", re.I)


def mentions(text: str, words) -> bool:
    for sent in re.split(r"(?<=[.;\n])\s+", text or ""):
        if any(re.search(rf"\b{w}\b", sent, re.I) for w in words) and not NEG.search(sent):
            return True
    return False


def max_mm(text: str):
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm", text or "")]
    sizes += [float(x) * 10 for x in re.findall(r"(\d+(?:\.\d+)?)\s*cm", text or "")]
    return max(sizes) if sizes else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv); rows = []
    for f in sorted(a.in_dir.glob("*_nvreason.json")):
        d = json.loads(f.read_text()); ans = d.get("answers", {}); rep = d.get("report", "")
        rows.append({"case_id": d["case_id"],
                     "nvr_lung_nodule_flag": mentions(ans.get("lung_nodules", "") + " " + rep, ["nodule", "mass"]),
                     "nvr_lung_nodule_max_mm": max_mm(ans.get("lung_nodules", "")),
                     "nvr_lymphadenopathy_flag": mentions(ans.get("lymphadenopathy", "") + " " + rep, ["lymphadenopathy", "enlarged lymph", "enlarged node", "adenopathy"]),
                     "nvr_lymph_max_mm": max_mm(ans.get("lymphadenopathy", "")),
                     "nvr_thyroid_flag": mentions(ans.get("thyroid", ""), ["nodule", "mass", "goiter", "goitre", "enlarged"]),
                     "nvr_thyroid_max_mm": max_mm(ans.get("thyroid", "")),
                     "nvr_malignancy_yes": bool(re.match(r"\s*yes", ans.get("malignancy", ""), re.I)),
                     "nvr_seconds": d.get("seconds")})
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.to_string(index=False) if len(df) else "no reports"); return 0


if __name__ == "__main__":
    sys.exit(main())
