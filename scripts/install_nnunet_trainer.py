#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _resolve_nnunet_trainer_dir() -> Path:
    import nnunetv2  # type: ignore

    base = Path(nnunetv2.__path__[0])
    trainer_dir = base / "training" / "nnUNetTrainer"
    if not trainer_dir.exists():
        raise FileNotFoundError(f"nnUNet trainer directory not found: {trainer_dir}")
    return trainer_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a custom nnUNetv2 trainer into site-packages")
    parser.add_argument("--trainer", required=True, help="Path to trainer .py file")
    parser.add_argument("--force", action="store_true", help="Overwrite if trainer exists")
    args = parser.parse_args()

    trainer_path = Path(args.trainer).resolve()
    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer file not found: {trainer_path}")

    dest_dir = _resolve_nnunet_trainer_dir()
    dest_path = dest_dir / trainer_path.name

    if dest_path.exists() and not args.force:
        raise FileExistsError(f"Trainer already exists: {dest_path} (use --force to overwrite)")

    shutil.copy2(trainer_path, dest_path)
    print(f"Installed trainer: {dest_path}")


if __name__ == "__main__":
    main()
