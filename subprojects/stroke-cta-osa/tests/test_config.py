"""Config parsing and override behaviour."""

from pathlib import Path

import pytest
import yaml

from stroke_cta_osa.config import PipelineConfig, apply_overrides, load_config


def test_default_config_loads():
    cfg = PipelineConfig()
    assert cfg.hu.fat_hu_min == -190
    assert cfg.hu.fat_hu_max == -30
    assert cfg.hu.air_hu_max == -500
    assert cfg.airway.fallback_method == "threshold_connected_component"
    assert cfg.coverage.include_cervical_soft_tissues == "required"


def test_yaml_loader_roundtrip(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "hu": {"fat_hu_min": -200},
        "airway": {"fallback_method": "none"},
    }))
    cfg = load_config(cfg_path)
    assert cfg.hu.fat_hu_min == -200
    assert cfg.airway.fallback_method == "none"
    # Untouched defaults remain
    assert cfg.hu.fat_hu_max == -30


def test_apply_overrides_nested():
    cfg = PipelineConfig()
    new = apply_overrides(cfg, {"airway.morphology_closing_mm": 3.0,
                                 "output.save_masks": True})
    assert new.airway.morphology_closing_mm == 3.0
    assert new.output.save_masks is True
    # Original config is not mutated
    assert cfg.airway.morphology_closing_mm == 1.0


def test_config_hash_stable():
    a = PipelineConfig().hash()
    b = PipelineConfig().hash()
    assert a == b
    c = apply_overrides(PipelineConfig(), {"hu.fat_hu_min": -150.0}).hash()
    assert c != a


def test_load_config_missing_path_returns_defaults():
    cfg = load_config(Path("/nonexistent/path/to/cfg.yaml"))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.hu.fat_hu_min == -190
