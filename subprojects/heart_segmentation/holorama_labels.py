"""Label scheme of the HolOrama whole-heart CT model (14 structures + background).

Values come from the model's ``config.json`` (``external/holorama/config.json``); the snake_case
keys are this repository's file names (``structures/<key>.nii.gz``); display names and colours
follow HolOrama's "Default 1 - Colorful" CCTA preset so a mask looks the same here and in HolOrama.

Label 9 is "Precardial Fat" in config.json and "Pericardial Fat" in HolOrama's presets; it is the
fat *outside* the pericardium (paracardial / pericardial), label 10 the epicardial fat inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

__all__ = [
    "LABELS",
    "LABEL_BY_ID",
    "LABEL_BY_KEY",
    "GROUPS",
    "TOTALSEG_EQUIVALENT",
    "HOLORAMA_CONFIG_NAMES",
    "label_table",
    "check_against_model_config",
]


@dataclass(frozen=True)
class Label:
    id: int
    key: str  # snake_case, used for file names and CSV columns
    name: str  # display name
    holorama_name: str  # name in HolOrama's config.json
    color: str  # hex, HolOrama "Default 1 - Colorful" preset

    @property
    def rgb(self) -> tuple[float, float, float]:
        h = self.color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


LABELS: tuple[Label, ...] = (
    Label(1, "coronary_arteries", "Coronary arteries", "Coronary Artery", "#ff3c3c"),
    Label(2, "lv_myocardium", "LV myocardium", "LVM", "#3cdc3c"),
    Label(3, "left_atrium", "Left atrium", "LA", "#3c3cff"),
    Label(4, "left_ventricle", "Left ventricle", "LV", "#ffdc00"),
    Label(5, "right_atrium", "Right atrium", "RA", "#dc3cdc"),
    Label(6, "right_ventricle", "Right ventricle", "RV", "#00d2d2"),
    Label(7, "aorta", "Aorta", "Aorta", "#ff8c00"),
    Label(8, "pulmonary_arteries", "Pulmonary arteries", "Pulmonary Arteries", "#a03cff"),
    Label(9, "pericardial_fat", "Pericardial (paracardial) fat", "Precardial Fat", "#00b4ff"),
    Label(10, "epicardial_fat", "Epicardial fat", "Epicardial Fat", "#ff3c8c"),
    Label(11, "pulmonary_veins", "Pulmonary veins", "Pulmonary Vein", "#00c878"),
    Label(12, "superior_vena_cava", "Superior vena cava", "Superior Vena Cava", "#b4ff00"),
    Label(13, "inferior_vena_cava", "Inferior vena cava", "Inferior Vena Cava", "#ffb464"),
    Label(14, "left_atrial_appendage", "Left atrial appendage", "LA Appendage", "#8c8cff"),
)
LABEL_BY_ID: dict[int, Label] = {lab.id: lab for lab in LABELS}
LABEL_BY_KEY: dict[str, Label] = {lab.key: lab for lab in LABELS}

GROUPS: dict[str, tuple[str, ...]] = {
    "chambers": ("left_atrium", "left_ventricle", "right_atrium", "right_ventricle", "left_atrial_appendage"),
    "myocardium": ("lv_myocardium",),
    "great_vessels": ("aorta", "pulmonary_arteries", "pulmonary_veins", "superior_vena_cava", "inferior_vena_cava"),
    "coronaries": ("coronary_arteries",),
    "fat": ("pericardial_fat", "epicardial_fat"),
    # everything that is contrast-filled blood on a CTA
    "blood_pool": (
        "left_atrium", "left_ventricle", "right_atrium", "right_ventricle", "left_atrial_appendage",
        "aorta", "pulmonary_arteries", "pulmonary_veins", "superior_vena_cava", "inferior_vena_cava",
        "coronary_arteries",
    ),
}

# TotalSegmentator (total / heartchambers_highres) structures that cover the same anatomy, for QC
# cross-checks. TotalSegmentator's LA includes the appendage; HolOrama separates it.
TOTALSEG_EQUIVALENT: dict[str, tuple[str, ...]] = {
    "left_atrium": ("heart_atrium_left", "atrial_appendage_left"),
    "left_ventricle": ("heart_ventricle_left",),
    "lv_myocardium": ("heart_myocardium",),
    "right_atrium": ("heart_atrium_right",),
    "right_ventricle": ("heart_ventricle_right",),
    "aorta": ("aorta",),
    "pulmonary_arteries": ("pulmonary_artery",),
    "pulmonary_veins": ("pulmonary_vein",),
    "superior_vena_cava": ("superior_vena_cava",),
    "inferior_vena_cava": ("inferior_vena_cava",),
    "left_atrial_appendage": ("atrial_appendage_left",),
}

HOLORAMA_CONFIG_NAMES: dict[str, int] = {"background": 0, **{lab.holorama_name: lab.id for lab in LABELS}}


def label_table() -> list[dict]:
    """The scheme as plain dicts (for JSON sidecars and Slicer)."""
    return [asdict(lab) for lab in LABELS]


def check_against_model_config(config_labels: dict[str, int]) -> None:
    """Raise if the model's config.json label map disagrees with this table (a different model
    version would silently mislabel every structure)."""
    got = {str(k): int(v) for k, v in config_labels.items()}
    if got != HOLORAMA_CONFIG_NAMES:
        missing = sorted(set(HOLORAMA_CONFIG_NAMES) - set(got))
        extra = sorted(set(got) - set(HOLORAMA_CONFIG_NAMES))
        changed = sorted(k for k in set(got) & set(HOLORAMA_CONFIG_NAMES) if got[k] != HOLORAMA_CONFIG_NAMES[k])
        raise ValueError(
            "The model's label map does not match holorama_labels.LABELS "
            f"(missing {missing}, extra {extra}, changed {changed}); update holorama_labels.py for this model."
        )
