"""Fat-compartment ontology: tiering, proxy labels, mask gating."""

import pytest

from stroke_cta_osa import fat_ontology as fo
from stroke_cta_osa.evidence_registry import EvidenceTier


def test_compartments_nonempty():
    assert len(fo.all_compartments()) > 15


def test_every_compartment_has_tier_and_proxy_label():
    for c in fo.all_compartments():
        assert isinstance(c.evidence_tier, EvidenceTier)
        assert c.true_anatomic_vs_proxy in {"anatomic", "proxy"}
        assert c.confidence_field  # non-empty


def test_tier1_core_compartments_present():
    t1 = {c.key for c in fo.compartments_for_tier(EvidenceTier.TIER_1_CORE_OSA_BACKED)}
    for key in ("cervical_total", "parapharyngeal_fat_pad_left",
                "parapharyngeal_fat_pad_retroglossal",
                "pharyngeal_airway_adjacent_fat"):
        assert key in t1, key


def test_supraplatysmal_is_proxy():
    c = fo.compartment("supraplatysmal_proxy_fat")
    assert c is not None
    assert c.true_anatomic_vs_proxy == "proxy"
    assert "platysma_mask" in c.optional_masks


def test_c5_and_pericarotid_are_tier3():
    for key in ("c5_nat_subcutaneous", "c5_nat_perivertebral", "pericarotid_fat"):
        assert fo.tier_for_compartment(key) == EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR


def test_engineered_ratios_are_tier4():
    assert fo.tier_for_compartment("engineered_fat_ratios") == \
        EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY


def test_confidence_scale():
    # required mask missing -> "missing"
    assert fo.confidence_for_masks(("airway_mask",), set(), is_proxy=False) == "missing"
    # proxy -> "low"
    assert fo.confidence_for_masks((), set(), is_proxy=True) == "low"
    # anatomic with required mask present -> "high"
    assert fo.confidence_for_masks(("airway_mask",), {"airway_mask"},
                                   is_proxy=False) == "high"
    # anatomic, no required masks -> "moderate"
    assert fo.confidence_for_masks((), set(), is_proxy=False) == "moderate"


def test_pericarotid_requires_carotid_mask():
    c = fo.compartment("pericarotid_fat")
    assert "carotid_mask" in c.required_masks
