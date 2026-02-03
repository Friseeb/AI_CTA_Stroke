#!/usr/bin/env python3
"""
Download TopCoW 2024 CLAIM weights from Zenodo and prepare a local models layout.

This script pulls all files from a Zenodo record, extracts archives, and attempts
to locate:
  - yolo-cow-detection.pt
  - topcow-claim-models (nnUNet folder)
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request


ZENODO_RECORD_ID = 14191592
ZENODO_API = "https://zenodo.org/api/records"


def _download(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:  # noqa: S310
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def _extract_if_archive(path: Path, extract_dir: Path) -> None:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        return
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz" or path.suffix == ".tar":
        with tarfile.open(path, "r:*") as tf:
            tf.extractall(extract_dir)


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = list(root.rglob(pattern))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download TopCoW CLAIM weights from Zenodo")
    p.add_argument("--output", required=True, help="Output directory for downloads")
    p.add_argument("--record", type=int, default=ZENODO_RECORD_ID, help="Zenodo record ID")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist")
    p.add_argument("--extract", action="store_true", default=True, help="Extract archives after download")
    p.add_argument("--no-extract", dest="extract", action="store_false", help="Disable archive extraction")
    p.add_argument("--prepare-layout", action="store_true", default=True, help="Create models/ layout with expected names")
    p.add_argument("--no-prepare-layout", dest="prepare_layout", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"{ZENODO_API}/{args.record}"
    print(f"Fetching Zenodo record {args.record}...")
    with urlopen(api_url) as resp:  # noqa: S310
        metadata = json.loads(resp.read().decode("utf-8"))

    files = metadata.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo record.")

    for entry in files:
        key = entry.get("key")
        links = entry.get("links", {})
        url = links.get("self") or links.get("download")
        if not key or not url:
            continue
        dest = output_dir / key
        if dest.exists() and not args.force:
            print(f"Skipping existing: {dest.name}")
        else:
            print(f"Downloading {key}...")
            _download(url, dest)
        if args.extract:
            _extract_if_archive(dest, output_dir)

    yolo_path = _find_first(output_dir, "yolo-cow-detection.pt")
    nnunet_dir = _find_first(output_dir, "topcow-claim-models")

    if args.prepare_layout:
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        if yolo_path:
            target = models_dir / "yolo-cow-detection.pt"
            if not target.exists():
                shutil.copy2(yolo_path, target)
            print(f"YOLO model: {target}")
        else:
            print("⚠ Could not find yolo-cow-detection.pt")
        if nnunet_dir and nnunet_dir.is_dir():
            target_dir = models_dir / "topcow-claim-models"
            if not target_dir.exists():
                shutil.copytree(nnunet_dir, target_dir, dirs_exist_ok=True)
            print(f"nnUNet models: {target_dir}")
        else:
            print("⚠ Could not find topcow-claim-models directory")
    else:
        if yolo_path:
            print(f"YOLO model: {yolo_path}")
        if nnunet_dir:
            print(f"nnUNet models: {nnunet_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
