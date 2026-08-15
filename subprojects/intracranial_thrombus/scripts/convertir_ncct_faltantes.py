"""
Convierte los NCCT que faltaban de la cohorte SLAO, desde el DICOM original.

Se usa dcm2niix v1.0.20220720 (/usr/bin/dcm2niix), ANTERIOR al bug de
v1.0.20250505 que escribe SliceThickness en vez del espaciado real. Aun asi se
verifica el spacing contra el DICOM antes de aceptar cada archivo.

Series elegidas (todas 'Vol. 0.5', la misma descripcion usada como NCCT en los
otros 86 casos de SLAOEVT):
    sub-27, sub-33, sub-62, sub-131  -> unica candidata, sin ambiguedad
    sub-58   -> tenia tambien 'NC NC 0.5' (270mm, cabeza+cuello); se elige
                'Vol. 0.5' (156mm, solo cabeza) por coherencia con el resto
                [decision del investigador]
    sub-176  -> tenia DOS series 'Vol. 0.5' del mismo estudio separadas 42 s
                (num 5 y num 20, 501 cortes cada una: dos reconstrucciones de
                la misma adquisicion). Se toma la num 5, la primera.

NO se convierten:
    sub-318  -> no tiene ninguna serie NCCT fina (solo bone 1.0 / sag / LCD 2.5)
    sub-1095 -> excluido del analisis (ver dedup_decisiones.EXCLUIR)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, SLAOEVT, SLAODICOM,
                   EXPORT_ROOT, NCCT_RAW, NCCT_REG, OUTPUT, V2, DCM2NIIX,
                   SLAO_FRESH, SLAOBIDS)
import shutil
import subprocess
import sys
import tempfile
import warnings

import nibabel as nib
import numpy as np
import pydicom

warnings.filterwarnings("ignore")

DICOM = SLAODICOM

# sub_id -> (SeriesNumber elegida, carpeta destino)
CASOS = {
    "27":  ("5", SLAO_FRESH / "sub-27"),
    "33":  ("5", SLAO_FRESH / "sub-33"),
    "58":  ("6", SLAO_FRESH / "sub-58"),
    "62":  ("6",  SLAOBIDS),
    "131":  ("5", SLAO_FRESH / "sub-131"),
    "176":  ("5", SLAO_FRESH / "sub-176"),
}


def spacing_real(files):
    """Espaciado Z real: extent/(n-1) sobre las posiciones unicas, deduplicando
    por SOPInstanceUID (mismo metodo verificado en el resto del proyecto)."""
    zs = {}
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            continue
        zs[str(getattr(ds, "SOPInstanceUID", str(f)))] = float(ipp[2])
    v = sorted(zs.values())
    if len(v) < 3:
        return None, len(v)
    return abs(v[-1] - v[0]) / (len(v) - 1), len(v)


for sub, (serie, destino) in CASOS.items():
    print(f"\n=== sub-{sub} (serie {serie}) ===", flush=True)
    src = DICOM / sub
    if not src.exists():
        print("  SIN DICOM, se omite")
        continue

    files = []
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", None) != "CT":
            continue
        if str(getattr(ds, "SeriesNumber", "")) != serie:
            continue
        if str(getattr(ds, "SeriesDescription", "")).strip() not in ("Vol. 0.5", "NC NC 0.5"):
            continue
        files.append(f)

    if not files:
        print(f"  no se encontraron archivos de la serie {serie}")
        continue

    sp_dcm, n = spacing_real(files)
    print(f"  {len(files)} archivos, {n} cortes unicos, spacing real DICOM = {sp_dcm:.4f} mm")

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "dcm"
        stage.mkdir()
        for i, f in enumerate(files):
            shutil.copy2(f, stage / f"{i:05d}.dcm")
        out = Path(td) / "out"
        out.mkdir()
        r = subprocess.run([DCM2NIIX, "-z", "y", "-f", f"sub-{sub}_acq-NCCT_ct",
                            "-o", str(out), str(stage)],
                           capture_output=True, text=True)
        nii = list(out.glob("*.nii.gz"))
        if not nii:
            print(f"  dcm2niix no produjo NIfTI. stderr: {r.stderr[-300:]}")
            continue
        nii = sorted(nii, key=lambda p: p.stat().st_size)[-1]

        img = nib.load(str(nii))
        sp_nii = float(img.header.get_zooms()[2])
        ax = "".join(nib.aff2axcodes(img.affine))
        ok = abs(sp_nii - sp_dcm) < 0.02
        print(f"  NIfTI: shape={img.shape} spacingZ={sp_nii:.4f} axcodes={ax} "
              f"-> {'OK' if ok else '*** DISCREPA con el DICOM ***'}")
        if not ok:
            print("  NO se copia: revisar manualmente")
            continue

        dst_dir = Path(destino)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"sub-{sub}_acq-NCCT_ct.nii.gz"
        shutil.copy2(nii, dst)
        js = nii.with_suffix("").with_suffix(".json")
        if js.exists():
            shutil.copy2(js, dst_dir / f"sub-{sub}_acq-NCCT_ct.json")
        print(f"  -> {dst}  ({dst.stat().st_size/1e6:.0f} MB)")
