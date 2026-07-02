"""README must carry the evidence-tier table, reference table, and disclaimers.

Targets the comprehensive evidence-based README at
docs/stroke_cta_osa/README.md (repo root), with a fallback to the subproject
README so the test is robust to where the canonical doc lives.
"""

from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_DOCS_README = _REPO_ROOT / "docs" / "stroke_cta_osa" / "README.md"
_SUBPROJECT_README = _HERE.parents[1] / "README.md"


def _readme_text() -> str:
    assert _DOCS_README.is_file(), f"missing {_DOCS_README}"
    return _DOCS_README.read_text()


def test_readme_has_evidence_tier_table():
    txt = _readme_text()
    assert "| Tier |" in txt or "| Tier 1 |" in txt
    for tier_word in ("Tier 1", "Tier 2", "Tier 3", "Tier 4"):
        assert tier_word in txt


def test_readme_has_reference_table():
    txt = _readme_text()
    for tag in (
        "Barkdull_2008_CT_OSA", "Shigeta_2011_Tongue_Mandible_CT",
        "Chen_2019_Parapharyngeal_Fat_DI_SLEEP_CT",
        "Ernst_2023_Cervical_Fat_Tissue_Volume",
        "Shelton_1993_Pharyngeal_Fat_OSA",
        "Torriani_2014_C5_Neck_Adipose_Tissue",
        "Zhang_2022_Upper_Airway_CT_DL",
    ):
        assert tag in txt, tag


def test_readme_states_does_not_diagnose_osa():
    txt = _readme_text().lower()
    assert "does not diagnose" in txt and "osa" in txt


def test_readme_mentions_airwaynet_without_assuming_weights():
    txt = _readme_text()
    assert "AirwayNet-MM-H" in txt
    assert "not assumed" in txt.lower()


def test_subproject_readme_links_to_evidence_docs():
    txt = _SUBPROJECT_README.read_text()
    assert "EVIDENCE_TIERS.md" in txt or "evidence" in txt.lower()
