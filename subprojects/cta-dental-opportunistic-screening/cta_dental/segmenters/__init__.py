"""Segmenter backends for dental/oral structures."""

from .base import BaseSegmenter, SegmentationResult
from .totalsegmentator import TotalSegmentatorTeethSegmenter
from .dentalsegmentator import DentalSegmentatorSegmenter
from .oralseg import OralSegSegmenter
from .rail import RAILSegmenter

__all__ = [
    "BaseSegmenter",
    "SegmentationResult",
    "TotalSegmentatorTeethSegmenter",
    "DentalSegmentatorSegmenter",
    "OralSegSegmenter",
    "RAILSegmenter",
]


def get_segmenter(name: str) -> BaseSegmenter:
    mapping = {
        "totalseg_teeth": TotalSegmentatorTeethSegmenter,
        "dentalsegmentator": DentalSegmentatorSegmenter,
        "oralseg": OralSegSegmenter,
        "rail": RAILSegmenter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown segmenter: {name!r}. Choose from {list(mapping)}")
    return mapping[name]()
