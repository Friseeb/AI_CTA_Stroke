#!/usr/bin/env python3
"""Create a vertebral review manifest from existing subject NIfTI files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDS = (
    "case_id",
    "cta_path",
    "label_path",
    "foramen_prior_path",
    "reviewer_id",
    "review_status",
    "notes",
)


def case_id_from_path(path: Path) -> str | None:
    match = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return match.group(1) if match else None


def is_subject_cta(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".nii.gz"):
        return False
    if "_mask" in name or "_defaced" in name:
        return False
    if "seg" in name or "label" in name or "prior" in name:
        return False
    return bool(re.search(r"sub-[a-z0-9]+(_0000|_acq-cta_ct)?\.nii\.gz$", name))


def iter_subject_ctas(roots: list[Path]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.nii.gz"):
            if is_subject_cta(path):
                case_id = case_id_from_path(path)
                if case_id and case_id not in seen:
                    seen.add(case_id)
                    paths.append(path.absolute())
    return sorted(paths, key=lambda p: case_id_from_path(p) or p.name)


def find_candidate(roots: list[Path], case_id: str, include_terms: tuple[str, ...], exclude_terms: tuple[str, ...] = ()) -> str:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(f"*{case_id}*.nii.gz"):
            name = path.name.lower()
            if all(term in name for term in include_terms) and not any(term in name for term in exclude_terms):
                candidates.append(path.absolute())
    if not candidates:
        return ""
    return str(sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0])


def build_rows(cta_paths: list[Path], label_roots: list[Path], foramen_roots: list[Path]) -> list[dict[str, str]]:
    rows = []
    for cta_path in cta_paths:
        case_id = case_id_from_path(cta_path)
        if not case_id:
            continue
        rows.append(
            {
                "case_id": case_id,
                "cta_path": str(cta_path),
                "label_path": find_candidate(label_roots, case_id, ("vert",), ("vertebrae",)),
                "foramen_prior_path": find_candidate(foramen_roots, case_id, ("foramen",)),
                "reviewer_id": "",
                "review_status": "in_progress",
                "notes": "",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cta-root", action="append", type=Path, required=True, help="Root containing subject CTA NIfTI files")
    parser.add_argument("--label-root", action="append", type=Path, default=[], help="Optional root containing vertebral labels")
    parser.add_argument("--foramen-root", action="append", type=Path, default=[], help="Optional root containing foramen priors")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest CSV")
    args = parser.parse_args()

    ctas = iter_subject_ctas(args.cta_root)
    rows = build_rows(ctas, args.label_root, args.foramen_root)
    write_manifest(args.out, rows)
    print(f"Wrote {len(rows)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
