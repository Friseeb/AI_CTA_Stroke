# Patch: `convert_SLAAO_dicom_to_bids.py` → use dcm2niix (not SimpleITK) + record spacing

**Why:** `_convert_to_nifti` used `sitk.ImageSeriesReader`, which sets the Z
spacing from only the *first* slice gap. For series with non-uniform / overlapping
slices this yields a wrong uniform Z → the volume is **deformed along Z** (axial
fine), and `SliceThickness` / `SpacingBetweenSlices` were never written to the
sidecars, so the true geometry couldn't be recovered later. dcm2niix derives slice
geometry correctly and writes a rich BIDS sidecar.

Apply on the box where the converter runs (the office/Linux machine). Requires
`dcm2niix` on PATH (or set `DCM2NIIX=/path/to/dcm2niix`).

---

## 1) Imports — add `subprocess` and `tempfile`

```python
import os
import re
import subprocess   # ADD
import tempfile     # ADD
import traceback
```

## 2) Replace `_convert_to_nifti` (currently SimpleITK) with this dcm2niix version

```python
DCM2NIIX = os.environ.get("DCM2NIIX") or shutil.which("dcm2niix") or "dcm2niix"

_DCM2NIIX_VERSION = None
def _dcm2niix_version() -> str:
    global _DCM2NIIX_VERSION
    if _DCM2NIIX_VERSION is None:
        try:
            out = subprocess.run([DCM2NIIX, "--version"], capture_output=True, text=True).stdout
            _DCM2NIIX_VERSION = (out.strip().splitlines() or ["unknown"])[-1]
        except Exception:
            _DCM2NIIX_VERSION = "unknown"
    return _DCM2NIIX_VERSION


def _convert_to_nifti(file_list: List[str], out_path: Path) -> dict:
    """Convert a pre-selected, ordered list of DICOM files to NIfTI via dcm2niix.

    dcm2niix derives slice geometry correctly (unlike SimpleITK's
    ImageSeriesReader, which takes Z spacing from only the first slice gap).
    Returns a geometry dict parsed from dcm2niix's BIDS sidecar
    (SliceThickness / SpacingBetweenSlices / PixelSpacing) for our own sidecar.
    """
    work = Path(tempfile.mkdtemp(prefix="dcm2niix_"))
    src = work / "src"
    src.mkdir()
    try:
        # dcm2niix consumes a directory; hardlink (fast) the selected files in.
        for i, f in enumerate(file_list):
            link = src / f"{i:06d}.dcm"
            try:
                os.link(f, link)
            except OSError:
                shutil.copy2(f, link)
        outdir = work / "out"
        outdir.mkdir()
        proc = subprocess.run(
            [DCM2NIIX, "-z", "y", "-m", "y", "-b", "y", "-f", "vol_%s",
             "-o", str(outdir), str(src)],
            capture_output=True, text=True,
        )
        niis = list(outdir.glob("*.nii.gz"))
        if not niis:
            raise RuntimeError(
                f"dcm2niix produced no NIfTI (rc={proc.returncode}): {proc.stderr[-500:]}"
            )
        # The largest file is the merged full-resolution volume (vs derived/tilt variants).
        best = max(niis, key=lambda p: p.stat().st_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(best), str(out_path))
        geom: dict = {}
        js = best.with_suffix("").with_suffix(".json")
        if js.exists():
            meta = json.loads(js.read_text())
            for k in ("SliceThickness", "SpacingBetweenSlices", "PixelSpacing"):
                if k in meta:
                    geom[k] = meta[k]
        return geom
    finally:
        shutil.rmtree(work, ignore_errors=True)
```

## 3) Capture the returned geometry at the call site

In `_convert_one_series`, change:

```python
        _convert_to_nifti(ordered, tmp)
```
to:
```python
        geom = _convert_to_nifti(ordered, tmp)
```

## 4) Record the spacing + converter in the sidecar

In the `sidecar = { ... }` dict, replace the conversion-software lines:

```python
            "ConversionSoftware":     "SimpleITK",
            "ConversionSoftwareVersion": sitk.Version_VersionString(),
            "OutputSizeMB":           round(mb, 3),
```
with:
```python
            "ConversionSoftware":        "dcm2niix",
            "ConversionSoftwareVersion": _dcm2niix_version(),
            "SliceThickness":            geom.get("SliceThickness"),
            "SpacingBetweenSlices":      geom.get("SpacingBetweenSlices"),
            "PixelSpacing":              geom.get("PixelSpacing"),
            "OutputSizeMB":              round(mb, 3),
```

(The `import SimpleITK as sitk` line can stay — it's now unused by conversion but
harmless. Remove it only after confirming nothing else uses `sitk`.)

---

## QC: catch a bad Z conversion automatically (optional but recommended)

After writing each NIfTI, compare its Z spacing to dcm2niix's reported
`SpacingBetweenSlices` and warn on mismatch:

```python
        try:
            import nibabel as nib
            zz = float(nib.load(str(out_path)).header.get_zooms()[2])
            sbs = geom.get("SpacingBetweenSlices")
            if sbs and abs(float(sbs) - zz) > 0.02:
                warnings.warn(f"{out_path.name}: NIfTI Zspacing {zz} != DICOM SpacingBetweenSlices {sbs}")
        except Exception:
            pass
```

## Re-convert existing pilot cases

The 11 LAA pilot subjects can be re-converted directly (no full pipeline run):
```bash
bash scripts/reconvert_slao_pilot_on_linux.sh \
     /media/fridmans/Research13T/datasets/SLAODICOM  ./slao_pilot_fixed
# copy slao_pilot_fixed/ back to the Mac -> outputs/laa_pilot_fixed/
```
Then on the Mac: repoint sessions + `run_laa_pilot_candidates.py --force`.
