#!/usr/bin/env python3
"""
Compute HU statistics inside one or more masks.

Supports:
  - binary masks (.nii/.nii.gz) via --mask
  - labelmaps via --labelmap + --label-id

Outputs JSON and/or CSV with summary stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj)


def _mask_from_labelmap(labelmap: np.ndarray, label_id: int) -> np.ndarray:
    return labelmap == label_id


def _mean_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    if x.size == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = std / np.sqrt(x.size) if x.size > 0 else float("nan")
    z = 1.96  # ~95% for normal
    return (mean - z * se, mean + z * se)


def _bootstrap_ci(
    x: np.ndarray, stat: str = "median", n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n)):
        sample = rng.choice(x, size=x.size, replace=True)
        if stat == "median":
            stats.append(float(np.median(sample)))
        elif stat == "mean":
            stats.append(float(np.mean(sample)))
        else:
            raise ValueError(f"Unknown stat for bootstrap: {stat}")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def _summarize(values: np.ndarray, bootstrap: int, seed: int) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "mean_ci95_low": float("nan"),
            "mean_ci95_high": float("nan"),
            "median_ci95_low": float("nan"),
            "median_ci95_high": float("nan"),
        }

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    p5 = float(np.quantile(values, 0.05))
    p95 = float(np.quantile(values, 0.95))
    mean_ci = _mean_ci(values)
    if bootstrap > 0:
        med_ci = _bootstrap_ci(values, stat="median", n=bootstrap, seed=seed)
    else:
        med_ci = (float("nan"), float("nan"))

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p5": p5,
        "p95": p95,
        "mean_ci95_low": float(mean_ci[0]),
        "mean_ci95_high": float(mean_ci[1]),
        "median_ci95_low": float(med_ci[0]),
        "median_ci95_high": float(med_ci[1]),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute HU stats inside CTA masks")
    p.add_argument("--cta", required=True, help="Input CTA NIfTI (HU intensities)")

    p.add_argument("--mask", action="append", default=[], help="Binary mask NIfTI (repeatable)")
    p.add_argument("--label", action="append", default=[], help="Label name for the last --mask")

    p.add_argument("--labelmap", action="append", default=[], help="Labelmap NIfTI (repeatable)")
    p.add_argument("--label-id", action="append", default=[], type=int, help="Label id for last --labelmap")
    p.add_argument("--labelmap-name", action="append", default=[], help="Label name for last --labelmap")

    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples for median CI (0 disables)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap")

    p.add_argument("--output-json", default=None, help="Write JSON summary")
    p.add_argument("--output-csv", default=None, help="Write CSV summary")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cta_path = Path(args.cta)
    if not cta_path.exists():
        raise FileNotFoundError(f"CTA not found: {cta_path}")

    cta = _load_nifti(cta_path).astype(np.float32)

    rows: List[Dict[str, float]] = []

    # Binary masks
    for idx, mask_path_str in enumerate(args.mask):
        mask_path = Path(mask_path_str)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = _load_nifti(mask_path) > 0
        label = args.label[idx] if idx < len(args.label) else mask_path.stem
        values = cta[mask]
        stats = _summarize(values, args.bootstrap, args.seed)
        stats["label"] = label
        stats["source"] = str(mask_path)
        rows.append(stats)

    # Labelmaps
    for idx, labelmap_path_str in enumerate(args.labelmap):
        labelmap_path = Path(labelmap_path_str)
        if not labelmap_path.exists():
            raise FileNotFoundError(f"Labelmap not found: {labelmap_path}")
        if idx >= len(args.label_id):
            raise ValueError("Each --labelmap needs a corresponding --label-id")
        label_id = int(args.label_id[idx])
        label = args.labelmap_name[idx] if idx < len(args.labelmap_name) else f"label_{label_id}"
        labelmap = _load_nifti(labelmap_path)
        mask = _mask_from_labelmap(labelmap, label_id)
        values = cta[mask]
        stats = _summarize(values, args.bootstrap, args.seed)
        stats["label"] = label
        stats["source"] = f"{labelmap_path}#id={label_id}"
        rows.append(stats)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"cta": str(cta_path), "stats": rows}, indent=2))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        # stable header order
        keys = [
            "label",
            "source",
            "n",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "q1",
            "q3",
            "iqr",
            "p5",
            "p95",
            "mean_ci95_low",
            "mean_ci95_high",
            "median_ci95_low",
            "median_ci95_high",
        ]
        lines = [",".join(keys)]
        for row in rows:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
        out.write_text("\n".join(lines))

    # Always print a short summary
    for row in rows:
        print(f"{row['label']}: n={row['n']}, mean={row['mean']:.2f}, median={row['median']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
