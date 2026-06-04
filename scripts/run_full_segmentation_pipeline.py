#!/usr/bin/env python3
"""
End-to-end wrapper: DICOM -> NIfTI -> deface -> segment (TS/TopCoW/NV/NUDF) -> merged labelmap.

This script orchestrates multiple tools/environments. It does not install anything.
Use the --*-env options to route steps through the correct conda envs.

Outputs (default under --output-dir):
  - nifti/         (DICOM -> NIfTI)
  - defaced/       (defaced NIfTI)
  - totalseg_total/
  - totalseg_headneck_bones_vessels/
  - totalseg_heartchambers_highres/
  - topcow/
  - nv_segment_ct_laa/
  - cardiac_ct_explorer/   (NUDF output)
  - labels_all.nii.gz + labels_all.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], env_name: str | None = None) -> None:
    if env_name:
        cmd = ["conda", "run", "-n", env_name] + cmd
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _find_first_nifti(folder: Path) -> Path:
    for ext in ("*.nii.gz", "*.nii"):
        matches = sorted(folder.glob(ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No NIfTI found in {folder}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full CTA segmentation wrapper")
    p.add_argument("--dicom-dir", default=None, help="Input DICOM folder")
    p.add_argument("--input-nifti", default=None, help="Input NIfTI (skip DICOM conversion)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--case-id", default="cta_case", help="Base name for outputs")

    p.add_argument("--dcm2niix", default="dcm2niix", help="Path to dcm2niix")
    p.add_argument("--deface", action="store_true", help="Run defacing")
    p.add_argument("--skip-deface", action="store_true", help="Skip defacing")

    p.add_argument("--run-totalseg", action="store_true", help="Run TotalSegmentator tasks")
    p.add_argument("--skip-totalseg", action="store_true", help="Skip TotalSegmentator tasks")

    p.add_argument("--run-topcow", action="store_true", help="Run TopCoW (Circle of Willis)")
    p.add_argument("--skip-topcow", action="store_true", help="Skip TopCoW")
    p.add_argument("--topcow-yolo-model", default=None, help="YOLO model path for TopCoW")
    p.add_argument("--topcow-nnunet-model-dir", default=None, help="nnUNet model dir for TopCoW")

    p.add_argument("--run-nv", action="store_true", help="Run NV-Segment-CT LAA")
    p.add_argument("--skip-nv", action="store_true", help="Skip NV-Segment-CT LAA")
    p.add_argument("--run-nv-aorta", action="store_true", help="Run NV-Segment-CT aorta (label 6)")
    p.add_argument("--skip-nv-aorta", action="store_true", help="Skip NV-Segment-CT aorta")

    p.add_argument("--run-nudf", action="store_true", help="Run NUDF LAA (CardiacCTExplorer)")
    p.add_argument("--skip-nudf", action="store_true", help="Skip NUDF LAA")

    p.add_argument("--merge-labels", action="store_true", help="Build merged label map")
    p.add_argument("--skip-merge", action="store_true", help="Skip merged label map")

    # Env routing
    p.add_argument("--totalseg-env", default="totalseg-mac", help="Conda env for TotalSegmentator")
    p.add_argument("--topcow-env", default="topcow_claim", help="Conda env for TopCoW")
    p.add_argument("--nv-env", default="nv-segment-ct", help="Conda env for NV-Segment-CT")
    p.add_argument("--nudf-env", default="cardiac-ct-explorer", help="Conda env for NUDF")
    p.add_argument("--merge-env", default="cardiac-ct-explorer", help="Conda env for merge step")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dicom_dir and not args.input_nifti:
        raise SystemExit("Provide --dicom-dir or --input-nifti")

    nifti_dir = out_dir / "nifti"
    deface_dir = out_dir / "defaced"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    deface_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: DICOM -> NIfTI (if needed)
    if args.input_nifti:
        input_nifti = Path(args.input_nifti)
    else:
        if shutil.which(args.dcm2niix) is None:
            raise SystemExit(f"dcm2niix not found: {args.dcm2niix}")
        dcm_dir = Path(args.dicom_dir)
        if not dcm_dir.exists():
            raise FileNotFoundError(f"DICOM dir not found: {dcm_dir}")
        _run([args.dcm2niix, "-z", "y", "-o", str(nifti_dir), "-f", args.case_id, str(dcm_dir)])
        input_nifti = _find_first_nifti(nifti_dir)

    # Step 2: Deface
    if not args.skip_deface and (args.deface or args.dicom_dir):
        defaced_path = deface_dir / f"{args.case_id}_defaced.nii.gz"
        _run(
            [
                "python",
                str(Path(__file__).parent / "deface_cta_simple.py"),
                "--input",
                str(input_nifti),
                "--output",
                str(defaced_path),
            ]
        )
    else:
        defaced_path = input_nifti

    # Step 3: TotalSegmentator tasks
    totalseg_total = out_dir / "totalseg_total"
    totalseg_headneck = out_dir / "totalseg_headneck_bones_vessels"
    totalseg_heart = out_dir / "totalseg_heartchambers_highres"
    if args.run_totalseg and not args.skip_totalseg:
        _run(
            [
                "python",
                str(Path(__file__).parent / "segment_external_models.py"),
                "--input",
                str(defaced_path),
                "--output",
                str(out_dir),
                "--totalseg-task",
                "total",
                "--totalseg-task",
                "headneck_bones_vessels",
                "--totalseg-task",
                "heartchambers_highres",
                "--totalseg-fullres",
            ],
            env_name=args.totalseg_env,
        )

    # Step 4: TopCoW
    topcow_dir = out_dir / "topcow"
    topcow_seg = topcow_dir / f"{args.case_id}_topcow_seg.nii.gz"
    if args.run_topcow and not args.skip_topcow:
        if not args.topcow_yolo_model or not args.topcow_nnunet_model_dir:
            raise SystemExit("TopCoW requires --topcow-yolo-model and --topcow-nnunet-model-dir")
        _run(
            [
                "python",
                str(Path(__file__).parent / "run_topcow_claim.py"),
                "--input",
                str(defaced_path),
                "--output",
                str(topcow_dir),
                "--yolo-model",
                str(args.topcow_yolo_model),
                "--nnunet-model-dir",
                str(args.topcow_nnunet_model_dir),
                "--labels-json",
                str(topcow_dir / "topcow_labels.json"),
            ],
            env_name=args.topcow_env,
        )
        # try to locate produced seg if name differs
        if not topcow_seg.exists():
            segs = sorted(topcow_dir.glob("*_topcow_seg.nii.gz"))
            if segs:
                topcow_seg = segs[0]

    # Step 5: NV LAA
    nv_out = out_dir / "nv_segment_ct_laa"
    nv_out.mkdir(parents=True, exist_ok=True)
    nv_laa = nv_out / f"{args.case_id}_laa108.nii.gz"
    if args.run_nv and not args.skip_nv:
        _run(
            [
                "python",
                str(Path(__file__).parent / "run_nv_segment_ct_laa.py"),
                "--input",
                str(defaced_path),
                "--output",
                str(nv_laa),
                "--model-dir",
                str(Path(__file__).parent.parent / "external" / "nv_segment_ct"),
                "--device",
                "auto",
            ],
            env_name=args.nv_env,
        )

    # Step 5b: NV aorta
    nv_aorta_out = out_dir / "nv_segment_ct_aorta"
    nv_aorta_out.mkdir(parents=True, exist_ok=True)
    nv_aorta = nv_aorta_out / f"{args.case_id}_aorta6.nii.gz"
    if args.run_nv_aorta and not args.skip_nv_aorta:
        _run(
            [
                "python",
                str(Path(__file__).parent / "run_nv_segment_ct_laa.py"),
                "--input",
                str(defaced_path),
                "--output",
                str(nv_aorta),
                "--label-id",
                "6",
                "--model-dir",
                str(Path(__file__).parent.parent / "external" / "nv_segment_ct"),
                "--device",
                "auto",
            ],
            env_name=args.nv_env,
        )

    # Step 6: NUDF LAA (CardiacCTExplorer)
    nudf_dir = out_dir / "cardiac_ct_explorer"
    nudf_laa = nudf_dir / f"{args.case_id}_laa8.nii.gz"
    if args.run_nudf and not args.skip_nudf:
        # Ensure CardiacCTExplorer expected TotalSegmentator layout
        ts_target = nudf_dir / "TotalSegmentator" / args.case_id
        ts_target.mkdir(parents=True, exist_ok=True)
        if totalseg_total.exists():
            shutil.copy2(totalseg_total / "total.nii.gz", ts_target / "total.nii.gz")
        if totalseg_heart.exists():
            shutil.copy2(totalseg_heart / "heartchambers_highres.nii.gz", ts_target / "heartchambers_highres.nii.gz")
        _run(
            [
                "python",
                str(Path(__file__).parent / "run_cardiac_ct_explorer_nudf_only.py"),
                "--input",
                str(defaced_path),
                "--output-dir",
                str(nudf_dir),
                "--laa-output",
                str(nudf_laa),
                "--device",
                "auto",
            ],
            env_name=args.nudf_env,
        )

    # Step 7: Merge all labels
    if args.merge_labels and not args.skip_merge:
        labels_out = out_dir / "labels_all.nii.gz"
        labels_json = out_dir / "labels_all.json"
        merge_cmd = [
            "python",
            str(Path(__file__).parent / "build_all_segmentations_labelmap.py"),
            "--reference",
            str(defaced_path),
            "--output",
            str(labels_out),
            "--labels-json",
            str(labels_json),
            "--totalseg-total",
            str(totalseg_total),
            "--totalseg-headneck",
            str(totalseg_headneck),
            "--totalseg-heartchambers",
            str(totalseg_heart),
            "--overwrite",
        ]
        if nv_laa.exists():
            merge_cmd += ["--laa-nv", str(nv_laa)]
        if nudf_laa.exists():
            merge_cmd += ["--laa-nudf", str(nudf_laa)]
        if topcow_seg.exists():
            merge_cmd += ["--topcow", str(topcow_seg)]
        if nv_aorta.exists():
            merge_cmd += ["--aorta-nv", str(nv_aorta)]
        _run(merge_cmd, env_name=args.merge_env)

    # Step 8: Export aorta candidates for comparison
    aorta_candidates = out_dir / "aorta_candidates"
    aorta_candidates.mkdir(parents=True, exist_ok=True)
    # TotalSegmentator total-task aorta
    total_aorta = totalseg_total / "aorta.nii.gz"
    if total_aorta.exists():
        shutil.copy2(total_aorta, aorta_candidates / "aorta_totalseg_total.nii.gz")
    # TotalSegmentator heartchambers_highres aorta
    heart_aorta = totalseg_heart / "aorta.nii.gz"
    if heart_aorta.exists():
        shutil.copy2(heart_aorta, aorta_candidates / "aorta_totalseg_heartchambers_highres.nii.gz")
    # NV aorta
    if nv_aorta.exists():
        shutil.copy2(nv_aorta, aorta_candidates / "aorta_nv_segment_ct.nii.gz")

    # Step 9: Build combined aorta candidate labelmap (single NIfTI)
    # Labels: 1=totalseg_total, 2=totalseg_heartchambers_highres, 3=nv_segment_ct
    try:
        import nibabel as nib
        import numpy as np

        reference_img = nib.load(str(defaced_path))
        reference_shape = reference_img.shape
        label_map = np.zeros(reference_shape, dtype=np.int16)

        candidate_specs = [
            (aorta_candidates / "aorta_totalseg_total.nii.gz", 1),
            (aorta_candidates / "aorta_totalseg_heartchambers_highres.nii.gz", 2),
            (aorta_candidates / "aorta_nv_segment_ct.nii.gz", 3),
        ]

        for path, label_id in candidate_specs:
            if not path.exists():
                continue
            img = nib.load(str(path))
            data = np.asarray(img.dataobj)
            if data.shape != reference_shape:
                print(f"  ⚠ Aorta candidate shape mismatch: {path} {data.shape} vs {reference_shape}")
                continue
            label_map[data > 0] = label_id

        combined_path = aorta_candidates / "aorta_candidates_labelmap.nii.gz"
        nib.save(nib.Nifti1Image(label_map, reference_img.affine, reference_img.header), str(combined_path))
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ Failed to build aorta candidate labelmap: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
