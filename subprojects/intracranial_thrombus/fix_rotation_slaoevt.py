"""
Aplica una rotacion IS (eje S-I / Z en RAS) permanente a las 4 series de un
sujeto de SLAOEVT_Nueva, horneandola en los archivos de disco.

Adaptado de fix_cta_rotation.py (que era solo para daylightbids) con dos
correcciones importantes:

  1. CENTRO DE ROTACION COMUN. El script original rotaba cada imagen alrededor
     de SU PROPIO centro. Como CTA, delays y NCCT tienen centros fisicos
     distintos (distinto FOV y numero de cortes), eso las habria DESALINEADO
     entre si. Aqui se calcula el centro una sola vez sobre el CTA y se usa
     ese mismo punto fisico para las 4 series.

  2. Escritura atomica (tmp + os.replace) y verificacion posterior, igual que
     el resto de reparaciones de esta cohorte.

Por que en disco y no con un transform en Slicer: el modulo de guardado
(CTAThrombusFinalizeSOP) exporta la mascara SIEMPRE en la rejilla del nodo CTA
y fuerza su IJKToRAS. Si se endurece un transform sobre el CTA dentro de
Slicer, la mascara se guardaria en un espacio que ya no coincide con el
archivo .nii.gz de disco, que es el que luego usa la extraccion de features.

Uso:
    python fix_rotation_slaoevt.py <sub_id> <angulo_grados> [--dry-run]

Ejemplo:
    python fix_rotation_slaoevt.py sub-522 -12.5
    python fix_rotation_slaoevt.py sub-522 -12.5 --dry-run   # solo simula

Los originales se respaldan como *_prerot.nii.gz (NO se toca el backup
*_original.nii.gz de la reparacion de spacing/signo, que debe conservarse).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, SLAOEVT, SLAODICOM,
                   EXPORT_ROOT, NCCT_RAW, NCCT_REG, OUTPUT, V2, DCM2NIIX)
import math
import os
import shutil
import sys

import SimpleITK as sitk

NUEVA = SLAOEVT
SERIES = ["CTA", "delay1", "delay2", "NCCT"]


def rotate_about(img: sitk.Image, angle_deg: float, center_lps) -> sitk.Image:
    """Rota la imagen sobre el eje IS alrededor de un punto fisico dado.

    SimpleITK trabaja en LPS; el eje IS de RAS es -Z en LPS, de ahi el signo
    invertido (mismo convenio que fix_cta_rotation.py)."""
    t = sitk.Euler3DTransform()
    t.SetCenter(center_lps)
    t.SetRotation(0.0, 0.0, -math.radians(angle_deg))  # rx, ry, rz
    return sitk.Resample(img, img, t, sitk.sitkLinear, -1024.0, img.GetPixelID())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sub_id = sys.argv[1]
    if not sub_id.startswith("sub-"):
        sub_id = f"sub-{sub_id}"
    angle = float(sys.argv[2])
    dry = "--dry-run" in sys.argv

    sub_dir = NUEVA / sub_id
    if not sub_dir.exists():
        print(f"No existe {sub_dir}")
        sys.exit(1)

    cta_path = sub_dir / f"{sub_id}_acq-CTA_ct.nii.gz"
    if not cta_path.exists():
        print(f"No existe el CTA: {cta_path}")
        sys.exit(1)

    # --- centro comun, calculado UNA vez sobre el CTA ---
    cta = sitk.ReadImage(str(cta_path))
    center_lps = cta.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in cta.GetSize()]
    )
    print(f"{sub_id} | rotacion IS {angle:+.2f} grados")
    print(f"  centro comun (LPS, del CTA): {[round(c, 2) for c in center_lps]}")
    if dry:
        print("  [DRY-RUN] no se escribe nada")

    for series in SERIES:
        path = sub_dir / f"{sub_id}_acq-{series}_ct.nii.gz"
        if not path.exists():
            print(f"  {series:7s} no existe, se omite")
            continue

        backup = path.with_name(path.name.replace(".nii.gz", "_prerot.nii.gz"))
        img = sitk.ReadImage(str(path))

        if dry:
            print(f"  {series:7s} size={img.GetSize()} -> se rotaria (backup: {backup.name})")
            continue

        if not backup.exists():
            shutil.copy2(path, backup)

        rotated = rotate_about(img, angle, center_lps)

        tmp = path.with_name(path.name.replace(".nii.gz", ".rot.tmp.nii.gz"))
        sitk.WriteImage(rotated, str(tmp), True)

        # verificacion antes de reemplazar
        chk = sitk.ReadImage(str(tmp))
        ok = (chk.GetSize() == img.GetSize()
              and all(abs(a - b) < 1e-6 for a, b in zip(chk.GetSpacing(), img.GetSpacing())))
        if not ok:
            print(f"  {series:7s} *** FALLO de verificacion, no se reemplaza (tmp: {tmp.name})")
            continue

        os.replace(str(tmp), str(path))
        print(f"  {series:7s} OK  (backup: {backup.name})")

    if not dry:
        print("\nHecho. Recarga el caso en Slicer para ver el resultado:")
        print(f"  load_case(<idx>)   # o next_case()")
        print("AVISO: si ya habias segmentado este sujeto, esa mascara quedo en")
        print("el espacio ANTERIOR a la rotacion y hay que rehacerla.")


if __name__ == "__main__":
    main()
