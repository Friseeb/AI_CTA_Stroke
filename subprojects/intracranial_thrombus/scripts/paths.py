"""
Rutas del subproyecto, configurables por variables de entorno.

Evita rutas absolutas fijadas al equipo del investigador: cada instalacion
apunta a las suyas exportando las variables correspondientes, o acepta los
valores por defecto si reproduce la estructura original.

Variables reconocidas:
    THROMBUS_REPO       raiz del repositorio            (por defecto: se deduce
                                                         de la ubicacion de este archivo)
    THROMBUS_DATASETS   raiz de los datasets de imagen  (BIDS, SLAOBIDS, ...)
    THROMBUS_SEG        raiz de las segmentaciones manuales
    THROMBUS_EXPORT     destino de las exportaciones a disco externo

Uso:
    from paths import REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, OUTPUT, V2
"""
import os
from pathlib import Path

# raiz del repo: .../subprojects/intracranial_thrombus/scripts/paths.py -> 3 niveles arriba
REPO = Path(os.environ.get("THROMBUS_REPO", Path(__file__).resolve().parents[3]))

DATASETS = Path(os.environ.get("THROMBUS_DATASETS", "/media/fridmans/Research13T/datasets"))
SEG_ROOT = Path(os.environ.get(
    "THROMBUS_SEG",
    "/home/fridmans/Desktop/INTRACRANEAL_THROMBUS/CTA_intracraneal_segmentation"))
EXPORT_ROOT = Path(os.environ.get("THROMBUS_EXPORT", "/media/fridmans/DICOM3"))

# subcarpetas derivadas
THROMBUS = REPO / "subprojects/intracranial_thrombus"
OUTPUT = THROMBUS / "output"
V2 = OUTPUT / "v2"
CONFIGS = THROMBUS / "configs"
COHORT_INPUTS = REPO / "cohort_generation/inputs"
COHORT_OUTPUTS = REPO / "cohort_generation/outputs"

# datasets de imagen
DAYLIGHT_BIDS = DATASETS / "daylightbids"
DAYLIGHT_BIDS_STD = DATASETS / "daylightbids_standard"
SLAOBIDS = DATASETS / "SLAOBIDS"
SLAO_FRESH = DATASETS / "SLAO_fresh_dcm2niix"
SLAOEVT = DATASETS / "SLAOEVT_Nueva"
SLAODICOM = DATASETS / "SLAODICOM"

# carpetas de segmentacion por cohorte
SEG_FOLDERS = {
    "DAYLIGHT": SEG_ROOT / "DAYLIGHT",
    "DAYLIGHT_STANDARD": SEG_ROOT / "DAYLIGHT_STANDARD",
    "SLAO": SEG_ROOT / "SLAO",
    "SLAOEVT": SLAOEVT / "thrombus_manual",
}
NCCT_RAW = SEG_ROOT / "DAYLIGHT_NCCT"
NCCT_REG = SEG_ROOT / "DAYLIGHT_NCCT_REG"

# herramientas externas
# v1.0.20220720 o anterior: NO tiene el bug que escribe SliceThickness en vez
# del espaciado real. Ver docs/NAMESPACE_WARNING.md y PROGRESO.md.
DCM2NIIX = os.environ.get("THROMBUS_DCM2NIIX", "/usr/bin/dcm2niix")
