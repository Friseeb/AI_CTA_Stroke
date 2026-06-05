import numpy as np

from aorta_cta_radiomics.encoders import (
    _encoder_backend_configs,
    build_wall_surface_patch_manifest,
    extract_encoder_features_from_masks,
)


def test_wall_surface_patch_manifest_samples_boundary_sectors_not_arch_shape():
    mask = np.zeros((12, 48, 48), dtype=bool)
    z, y, x = np.indices(mask.shape)
    ascending = ((x - 16) ** 2 + (y - 24) ** 2) <= 5**2
    descending = ((x - 32) ** 2 + (y - 24) ** 2) <= 5**2
    mask[ascending | descending] = True

    frame = build_wall_surface_patch_manifest(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        patch_size_mm_zyx=(8.0, 16.0, 16.0),
        axial_step_mm=4.0,
        angular_bins=8,
        max_patches=12,
    )

    assert not frame.empty
    assert frame["source"].eq("wall_surface_grid").all()
    assert frame["wall_component_rank_in_slice"].nunique() == 2
    assert frame["patch_size_y_mm"].eq(16.0).all()


def test_encoder_backend_configs_support_ensemble_and_disabled_entries():
    configs = _encoder_backend_configs(
        {
            "device": "cpu",
            "patch_sources": ["calcification_500HU"],
            "backends": [
                {"name": "tap_ct", "backend": "tap_ct_hf", "model_name": "fomofo/tap-ct-b-3d"},
                {"name": "voxelfm", "backend": "voxelfm_hf", "enabled": False},
            ],
        }
    )

    assert len(configs) == 1
    assert configs[0]["name"] == "tap_ct"
    assert configs[0]["device"] == "cpu"
    assert configs[0]["patch_sources"] == ["calcification_500HU"]


def test_unsupported_encoder_backend_returns_status_feature_without_crash():
    image = np.zeros((8, 16, 16), dtype=np.float32)
    source_mask = np.zeros_like(image, dtype=bool)
    source_mask[2:4, 6:8, 6:8] = True

    features, manifest = extract_encoder_features_from_masks(
        image=image,
        source_masks={"calcification_500HU": source_mask},
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        config={
            "encoders": {
                "enabled": True,
                "patch_sources": ["calcification_500HU"],
                "max_patches_per_source": 1,
                "patch_size_mm_zyx": [4, 4, 4],
                "backends": [{"name": "bad_backend", "backend": "not_real"}],
            }
        },
    )

    assert not manifest.empty
    assert features["feature_group"].eq("encoder_bad_backend_status").all()
    assert features["feature_name"].tolist() == ["extraction_error"]
