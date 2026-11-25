#!/usr/bin/env python3
"""Download the HolOrama whole-heart CT model into external/holorama/ (pinned commit, sha256-checked).

  python subprojects/heart_segmentation/fetch_holorama_weights.py            # download (~21 MB)
  python subprojects/heart_segmentation/fetch_holorama_weights.py --check    # verify what is there
  python subprojects/heart_segmentation/fetch_holorama_weights.py --from-clone <path/to/HolOrama>

Source: https://github.com/AI-in-Cardiovascular-Medicine/HolOrama (MIT License), the files under
src/segmentation/weights/ct_seg/ plus the LICENSE. A new HolOrama release with a retrained model must be
re-pinned here (COMMIT + SHA256) and checked against holorama_labels.LABELS before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE.parents[1] / "external" / "holorama"
REPO_URL = "https://github.com/AI-in-Cardiovascular-Medicine/HolOrama"
COMMIT = "e6ce0460249be20df058cd204df332f9e01aa594"  # 2026-11-20, HolOrama 0.21.0
RAW = "https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/HolOrama/{commit}/{path}"
# local name -> (path in the HolOrama repository, sha256)
FILES = {
    "ct_seg_u2net_small.onnx": (
        "src/segmentation/weights/ct_seg/ct_seg_u2net_small.onnx",
        "ffd4910db3890438353fe518f2afcb60418b36468f1b7da0456115d34a149240",
    ),
    "ct_seg_u2net_small_fp16.onnx": (
        "src/segmentation/weights/ct_seg/ct_seg_u2net_small_fp16.onnx",
        "361bbb743a9949d50acf47d73f65f7e437d029430d75499c654ad278bac16922",
    ),
    "config.json": (
        "src/segmentation/weights/ct_seg/config.json",
        "5ab5a782a66c84f617d92eb81ac0b0d882195734f006a46ad6854de28961c9b3",
    ),
    "LICENSE": ("LICENSE", "6d2b7e96a3fdd69765631e0ebc16f831633b96638aa24247c20fd7ca78cbcfdf"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(dest: Path) -> dict[str, str]:
    """name -> 'ok' | 'missing' | 'sha256 mismatch'."""
    out = {}
    for name, (_, expected) in FILES.items():
        p = dest / name
        out[name] = "missing" if not p.is_file() else ("ok" if sha256(p) == expected else "sha256 mismatch")
    return out


def fetch(dest: Path, from_clone: Path | None = None, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    state = check(dest)
    for name, (rel, expected) in FILES.items():
        target = dest / name
        if state[name] == "ok" and not force:
            print(f"{name}: present, sha256 ok")
            continue
        tmp = dest / (name + ".part")
        if from_clone is not None:
            src = Path(from_clone) / rel
            if not src.is_file():
                raise SystemExit(f"{src} not found in the clone")
            shutil.copyfile(src, tmp)
            origin = str(src)
        else:
            url = RAW.format(commit=COMMIT, path=rel)
            print(f"{name}: downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "ai-cta-stroke-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh)
            origin = url
        got = sha256(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{name}: sha256 mismatch ({got} != {expected}) from {origin}; not keeping the file")
        tmp.replace(target)
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB, sha256 ok")
    (dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "repository": REPO_URL,
                "commit": COMMIT,
                "license": "MIT",
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {n: {"path": rel, "sha256": sha} for n, (rel, sha) in FILES.items()},
                "citation": [
                    "Mohammadi Kazaj P et al. A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping. arXiv:2607.11287 (2026).",
                    "Stark AW et al. Automated intravascular ultrasound image processing ... the HolOrama software. Comput Methods Programs Biomed 2025;109065.",
                ],
            },
            indent=2,
        )
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--from-clone", type=Path, help="copy from a local HolOrama checkout instead of downloading")
    ap.add_argument("--check", action="store_true", help="only report the state of the files")
    ap.add_argument("--force", action="store_true", help="re-download files that are already ok")
    a = ap.parse_args(argv)
    if a.check:
        state = check(a.dest)
        for name, s in state.items():
            print(f"{name}: {s}")
        return 0 if all(s == "ok" for s in state.values()) else 1
    fetch(a.dest, a.from_clone, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
