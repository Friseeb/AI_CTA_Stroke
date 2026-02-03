"""Pipeline orchestrator: chains all stages of Antiga 2008 centerline extraction."""
from __future__ import annotations

# IMPORTANT: Import juliacall BEFORE torch to avoid segfault.
# Set JULIACALL_ENABLE=1 to opt in (disabled by default to avoid crashes).
# See: https://github.com/pytorch/pytorch/issues/78829
import os
if os.environ.get("JULIACALL_ENABLE") == "1":
    try:
        import juliacall  # noqa: F401
    except ImportError:
        pass

import copy
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

from .stage1_surface_extraction import extract_surface
from .stage2_extremal_points import detect_extremal_points
from .stage4_eikonal import extract_centerlines_via_eikonal
from .stage5_radius import compute_radii
from .stage6_bifurcations import detect_bifurcations
from .stage7_graph import build_centerline_graph, export_graph


class CenterlineExtractionPipeline:
    """Orchestrates multi-stage Antiga 2008 centerline extraction.

    Stages:
        1. Surface extraction (morphological cleaning)
        2. Extremal points detection (distance map maxima)
        3. Voronoi skeleton (implicit, distance map)
        4. Eikonal path tracing (gradient descent on distance map)
        5. Radius computation (trilinear interpolation)
        6. Bifurcation detection (tube containment)
        7. Graph construction (NetworkX export)
    """

    def __init__(
        self,
        nifti_path: str | Path,
        output_dir: str | Path | None = None,
        log_level: str = 'INFO',
        cta_settings: dict | None = None,
        device: str = 'auto',
    ):
        """Initialize pipeline.

        Parameters
        ----------
        nifti_path : str | Path
            Path to binary vessel mask (NIfTI)
        output_dir : str | Path, optional
            Directory for outputs; defaults to nifti_path parent / 'centerline_output'
        log_level : str
            Logging level
        cta_settings : dict, optional
            Only used when the input NIfTI looks like raw CTA intensities (not a binary mask).
            Accepted keys (defaults in parentheses):
            threshold_hu (150), max_hu (700), bone_hu (900), strip_boundary_bone (True),
            boundary_margin_mm (6.0), min_component_size (500), save_mask (False),
            mask_output_path (None → output_dir / 'cta_vessel_mask.nii.gz'),
            bone_mask_path (None), vessel_mask_path (None), apply_bone_mask_early (False)
        device : str
            'auto' (prefer MPS then CUDA), 'cuda', 'mps', or 'cpu'.
        """
        self.nifti_path = Path(nifti_path)

        if output_dir is None:
            output_dir = self.nifti_path.parent / 'centerline_output'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(self.output_dir / 'pipeline.log')
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        self.logger.info(f'Initialized pipeline for {self.nifti_path}')
        self.results = {}

        # CTA preprocessing defaults (only used when input is not already a binary mask)
        self.cta_settings = {
            'threshold_hu': 150,
            'max_hu': 700,
            'bone_hu': 900,
            'strip_boundary_bone': True,
            'boundary_margin_mm': 6.0,
            'min_component_size': 500,
            'save_mask': False,
            'mask_output_path': None,
            'bone_mask_path': None,
            'vessel_mask_path': None,
            'apply_bone_mask_early': False,
        }
        if cta_settings:
            self.cta_settings.update(cta_settings)

        self.device = self._select_device(device)
        self.logger.info(f'Compute device: {self.device}')

    def load_image(self) -> np.ndarray:
        """Load NIfTI image and extract binary mask.

        Returns
        -------
        np.ndarray
            3D binary array of vessel segmentation
        """
        self.logger.info(f'Loading image from {self.nifti_path}')
        img = nib.load(self.nifti_path)
        data = np.asarray(img.dataobj)

        if self._is_binary_mask(data):
            mask = data > 0
            self.logger.info('Detected binary mask input; using >0 voxels as vessel mask')
        else:
            self.logger.info('Input looks like CTA intensities; creating vessel mask via HU thresholding')
            mask = self._create_mask_from_cta(data, img) > 0

        self.logger.info(f'Image shape: {mask.shape}, voxel count: {mask.sum()}')
        return mask

    @staticmethod
    def _compute_crop_bounds(
        mask: np.ndarray,
        margin_vox: int | tuple[int, int, int] = 8,
    ) -> tuple[int, int, int, int, int, int]:
        """Compute crop bounds for a mask with a voxel margin."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])

        if isinstance(margin_vox, (tuple, list)) and len(margin_vox) == 3:
            mx, my, mz = [int(m) for m in margin_vox]
        else:
            mx = my = mz = int(margin_vox)

        x0 = max(int(coords[0].min()) - mx, 0)
        x1 = min(int(coords[0].max()) + mx + 1, mask.shape[0])
        y0 = max(int(coords[1].min()) - my, 0)
        y1 = min(int(coords[1].max()) + my + 1, mask.shape[1])
        z0 = max(int(coords[2].min()) - mz, 0)
        z1 = min(int(coords[2].max()) + mz + 1, mask.shape[2])

        return (x0, x1, y0, y1, z0, z1)

    @staticmethod
    def _crop_and_downsample(
        mask: np.ndarray,
        margin_vox: int | tuple[int, int, int] = 8,
        downsample_factor: int = 1,
    ) -> tuple[np.ndarray, tuple[int, int, int], int]:
        """Crop mask to bounding box + margin; optional strided downsample.

        Returns cropped/downsampled mask, offset (orig space), and scale (int factor).
        """
        def downsample_max_pool(vol: np.ndarray, ds: int) -> np.ndarray:
            if ds <= 1:
                return vol
            # Preserve thin vessels by max-pooling blocks instead of striding.
            pad_x = (-vol.shape[0]) % ds
            pad_y = (-vol.shape[1]) % ds
            pad_z = (-vol.shape[2]) % ds
            if pad_x or pad_y or pad_z:
                vol = np.pad(
                    vol,
                    ((0, pad_x), (0, pad_y), (0, pad_z)),
                    mode='constant',
                    constant_values=False,
                )
            sx, sy, sz = vol.shape
            vol = vol.reshape(sx // ds, ds, sy // ds, ds, sz // ds, ds)
            return vol.max(axis=(1, 3, 5))

        x0, x1, y0, y1, z0, z1 = CenterlineExtractionPipeline._compute_crop_bounds(
            mask, margin_vox
        )

        if x0 == 0 and x1 == mask.shape[0] and y0 == 0 and y1 == mask.shape[1] and z0 == 0 and z1 == mask.shape[2]:
            # Empty mask -> no crop
            if not mask.any():
                return mask, (0, 0, 0), 1

        cropped = mask[x0:x1, y0:y1, z0:z1]

        ds = max(1, int(downsample_factor))
        if ds > 1:
            cropped = downsample_max_pool(cropped.astype(bool), ds)

        return cropped, (x0, y0, z0), ds

    @staticmethod
    def _label_connected_components(
        mask: np.ndarray,
        min_component_size: int = 1,
    ) -> tuple[np.ndarray, int]:
        """Label connected components with optional size filtering."""
        labeled, num = label(mask)
        if num == 0:
            return np.zeros_like(mask, dtype=np.int32), 0

        min_component_size = int(min_component_size)
        if min_component_size <= 1:
            return labeled.astype(np.int32), int(num)

        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_component_size
        keep[0] = False
        kept_labels = np.where(keep)[0]

        relabeled = np.zeros_like(labeled, dtype=np.int32)
        for new_id, old_id in enumerate(kept_labels, start=1):
            relabeled[labeled == old_id] = new_id

        return relabeled, int(len(kept_labels))

    @staticmethod
    def _restore_labels_to_full(
        cropped_labels: np.ndarray,
        crop_bounds: tuple[int, int, int, int, int, int],
        scale: int,
        full_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Restore cropped/downsampled labels to full-resolution volume."""
        labels = cropped_labels
        if scale > 1:
            labels = np.repeat(labels, scale, axis=0)
            labels = np.repeat(labels, scale, axis=1)
            labels = np.repeat(labels, scale, axis=2)

        x0, x1, y0, y1, z0, z1 = crop_bounds
        crop_shape = (x1 - x0, y1 - y0, z1 - z0)
        labels = labels[:crop_shape[0], :crop_shape[1], :crop_shape[2]]

        full = np.zeros(full_shape, dtype=labels.dtype)
        full[x0:x1, y0:y1, z0:z1] = labels
        return full

    @staticmethod
    def _rescale_centerlines(centerlines: dict, offset: tuple[int, int, int], scale: int) -> dict:
        """Rescale centerline paths/radii back to original voxel space."""
        if scale == 1 and offset == (0, 0, 0):
            return centerlines

        offset_arr = np.array(offset, dtype=float)
        scale_f = float(scale)

        for seg_id, seg_data in centerlines.items():
            path = np.array(seg_data['path'], dtype=float)
            path = path * scale_f + offset_arr
            seg_data['path'] = path
            if len(path) > 1:
                diffs = np.diff(path, axis=0)
                seg_data['length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
            else:
                seg_data['length'] = 0.0
            if 'radii' in seg_data:
                seg_data['radii'] = np.array(seg_data['radii'], dtype=float) * scale_f
            centerlines[seg_id] = seg_data

        return centerlines

    def _select_device(self, device: str) -> str:
        if device in ('cuda', 'mps', 'cpu'):
            return device
        try:
            import torch

            if device == 'auto':
                if torch.backends.mps.is_available():
                    return 'mps'
                if torch.cuda.is_available():
                    return 'cuda'
                return 'cpu'
        except Exception:  # noqa: BLE001
            return 'cpu'
        return 'cpu'

    @staticmethod
    def _is_binary_mask(data: np.ndarray) -> bool:
        """Heuristic to decide if the NIfTI already contains a binary vessel mask."""
        finite = np.isfinite(data)
        if not finite.all():
            data = np.nan_to_num(data)
        unique_vals = np.unique(data)
        if unique_vals.size <= 3 and unique_vals.min() >= 0 and unique_vals.max() <= 1:
            return True
        return False

    def _create_mask_from_cta(self, data: np.ndarray, img: nib.Nifti1Image) -> np.ndarray:
        """Threshold and clean CTA intensities to produce a vessel mask."""
        cfg = self.cta_settings
        threshold_hu = cfg.get('threshold_hu', 150)
        max_hu = cfg.get('max_hu', 700)
        bone_hu = cfg.get('bone_hu', 900)
        strip_boundary_bone = cfg.get('strip_boundary_bone', True)
        boundary_margin_mm = cfg.get('boundary_margin_mm', 6.0)
        min_component_size = cfg.get('min_component_size', 500)
        bone_mask_path = cfg.get('bone_mask_path')
        vessel_mask_path = cfg.get('vessel_mask_path')
        apply_bone_mask_early = cfg.get('apply_bone_mask_early', False)

        data = np.nan_to_num(data)

        bandpass = data > threshold_hu
        if max_hu is not None:
            bandpass &= data < max_hu

        bone_mask_data = None

        if apply_bone_mask_early and bone_mask_path:
            bone_mask_img = nib.load(str(bone_mask_path))
            bone_mask_data = bone_mask_img.get_fdata() > 0
            if bone_mask_data.shape != data.shape:
                raise ValueError('Bone mask shape does not match CTA shape')
            bandpass &= ~bone_mask_data
            self.logger.info(
                'Applied bone mask early (warning: may exclude vertebral arteries in foramina)'
            )
        elif bone_mask_path:
            self.logger.info(
                'Bone mask provided but not applied early; using HU-only boundary stripping'
            )

        if vessel_mask_path:
            vessel_mask_img = nib.load(str(vessel_mask_path))
            vessel_mask_data = vessel_mask_img.get_fdata() > 0
            if vessel_mask_data.shape != data.shape:
                raise ValueError('Vessel mask shape does not match CTA shape')
            bandpass |= vessel_mask_data

        mask = bandpass.astype(np.uint8)

        if strip_boundary_bone:
            bone_mask = (data >= bone_hu).astype(np.uint8)
            self.logger.info(f'Stripping boundary bone using HU >= {bone_hu}')
            labeled_bone, _ = label(bone_mask)

            boundary_ids = set()
            faces = [
                labeled_bone[0, :, :], labeled_bone[-1, :, :],
                labeled_bone[:, 0, :], labeled_bone[:, -1, :],
                labeled_bone[:, :, 0], labeled_bone[:, :, -1],
            ]
            for face in faces:
                boundary_ids.update(np.unique(face))
            boundary_ids.discard(0)

            if boundary_margin_mm > 0:
                vx, vy, vz = img.header.get_zooms()
                margin_vx = max(1, int(round(boundary_margin_mm / min(vx, vy, vz))))
                boundary_shell = np.zeros_like(bone_mask, dtype=bool)
                boundary_shell[:margin_vx, :, :] = True
                boundary_shell[-margin_vx:, :, :] = True
                boundary_shell[:, :margin_vx, :] = True
                boundary_shell[:, -margin_vx:, :] = True
                boundary_shell[:, :, :margin_vx] = True
                boundary_shell[:, :, -margin_vx:] = True
                boundary_shell &= bone_mask.astype(bool)
                shell_ids = np.unique(labeled_bone[boundary_shell])
                shell_ids = set(shell_ids.tolist())
                shell_ids.discard(0)
                boundary_ids.update(shell_ids)

            if boundary_ids:
                boundary_bone = np.isin(labeled_bone, list(boundary_ids))
                mask &= ~boundary_bone

        labeled, _ = label(mask)
        component_sizes = np.bincount(labeled.ravel())
        large_components = np.where(component_sizes > min_component_size)[0]
        large_components = large_components[large_components > 0]
        mask_filtered = np.isin(labeled, large_components).astype(np.uint8)

        mask_clean = binary_erosion(mask_filtered, iterations=1)
        mask_clean = binary_dilation(mask_clean, iterations=1).astype(np.uint8)

        if cfg.get('save_mask', False):
            mask_path = cfg.get('mask_output_path')
            if mask_path is None:
                mask_path = self.output_dir / 'cta_vessel_mask.nii.gz'
            mask_path = Path(mask_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask_clean, img.affine, img.header), str(mask_path))
            self.logger.info(f'Saved CTA-derived mask to {mask_path}')

        return mask_clean

    def _save_intermediate(self, name: str, data: np.ndarray, is_mask: bool = True):
        """Save intermediate output as NIfTI for debugging/visualization."""
        if not self.save_intermediates:
            return
        out_path = self.output_dir / f'{name}.nii.gz'
        # Load original image for affine/header
        img = nib.load(self.nifti_path)
        if is_mask:
            data = data.astype(np.uint8)
        else:
            data = data.astype(np.float32)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out_path))
        self.logger.info(f'Saved intermediate: {out_path}')

    def _save_label_map(self, label_map: np.ndarray, output_path: Path):
        """Save labeled segmentation as NIfTI."""
        img = nib.load(self.nifti_path)
        label_map = label_map.astype(np.int32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(label_map, img.affine, img.header), str(output_path))
        self.logger.info(f'Saved label map: {output_path}')

    def run(self, **stage_kwargs) -> dict:
        """Execute full pipeline.

        Parameters
        ----------
        **stage_kwargs : dict
            Keyword arguments for individual stages
            (e.g., min_component_size=50, contact_distance_threshold=1.0)

            Special kwargs:
            - save_intermediates: bool, save .nii.gz at each stage (default False)
            - allow_cpu_edt: bool, allow CPU EDT fallback if GPU backend unavailable (default True)
            - save_label_map: bool, save connected-component label map (default False)
            - label_map_output_path: optional path for label map .nii.gz
            - label_map_source: 'stage1' (default) or 'input'
            - label_map_min_component_size: int, min component size for labels (default 1)

        Returns
        -------
        dict
            Pipeline results with keys: stage1, stage2, stage4, stage5, stage6, stage7
        """
        self.save_intermediates = stage_kwargs.pop('save_intermediates', False)
        save_label_map = stage_kwargs.pop('save_label_map', False)
        label_map_output_path = stage_kwargs.pop('label_map_output_path', None)
        label_map_source = stage_kwargs.pop('label_map_source', 'stage1')
        label_map_min_component_size = stage_kwargs.pop('label_map_min_component_size', 1)
        if label_map_min_component_size is None:
            label_map_min_component_size = 1

        try:
            # Load image
            mask = self.load_image()
            self.results['mask'] = mask

            # Stage 1: Surface extraction
            self.logger.info('Running Stage 1: Surface extraction')
            crop_margin = stage_kwargs.get('crop_margin_vox', 8)
            downsample_factor = stage_kwargs.get('downsample_factor', 1)
            crop_bounds = self._compute_crop_bounds(mask, crop_margin)
            edt_backend = stage_kwargs.get('edt_backend', 'auto')
            allow_cpu_edt = stage_kwargs.get('allow_cpu_edt', True)
            thick_component_max_radius = stage_kwargs.get('thick_component_max_radius', None)
            erosion_iterations = stage_kwargs.get('erosion_iterations', 1)
            dilation_iterations = stage_kwargs.get('dilation_iterations', 1)
            proc_mask, offset, scale = self._crop_and_downsample(
                mask, margin_vox=crop_margin, downsample_factor=downsample_factor
            )
            self.results['crop_offset'] = offset
            self.results['downsample_factor'] = scale
            self.logger.info(
                f'Mask crop to bbox+margin {crop_margin} → shape {proc_mask.shape}, offset {offset}, ds factor {scale}'
            )

            # Set GPU backend based on device selection
            # MPS uses Metal for EDT, CUDA uses CuPy
            if self.device == 'mps':
                effective_edt_backend = 'metal' if edt_backend == 'auto' else edt_backend
                effective_gpu_backend = None  # EDT via Metal, not CuPy
            elif self.device == 'cuda':
                effective_edt_backend = edt_backend
                effective_gpu_backend = 'cuda'
            else:
                effective_edt_backend = edt_backend
                effective_gpu_backend = None

            self.logger.info(f'GPU config: device={self.device}, edt_backend={effective_edt_backend}')

            stage1_result = extract_surface(
                proc_mask,
                min_component_size=stage_kwargs.get('min_component_size', 50),
                erosion_iterations=erosion_iterations,
                dilation_iterations=dilation_iterations,
                thick_component_max_radius=thick_component_max_radius,
                edt_backend=effective_edt_backend,
                gpu_backend=effective_gpu_backend,
                allow_cpu_edt=allow_cpu_edt,
            )
            self.results['stage1'] = stage1_result
            self.logger.info(
                f'Stage 1 complete: {stage1_result["num_components"]} components, '
                f'EDT backend: {stage1_result.get("distance_backend", "unknown")}'
            )

            if save_label_map:
                label_map_source = str(label_map_source).lower()
                if label_map_source not in ('stage1', 'input'):
                    raise ValueError(
                        "label_map_source must be 'stage1' or 'input'"
                    )

                if label_map_source == 'input':
                    label_map_full, num_labels = self._label_connected_components(
                        mask, label_map_min_component_size
                    )
                else:
                    cropped_labels, num_labels = self._label_connected_components(
                        stage1_result['cleaned_mask'], label_map_min_component_size
                    )
                    label_map_full = self._restore_labels_to_full(
                        cropped_labels, crop_bounds, scale, mask.shape
                    )

                label_map_path = (
                    Path(label_map_output_path)
                    if label_map_output_path is not None
                    else self.output_dir / 'segmentation_labels.nii.gz'
                )
                self._save_label_map(label_map_full, label_map_path)
                self.results['label_map'] = {
                    'path': str(label_map_path),
                    'num_labels': int(num_labels),
                    'source': label_map_source,
                }
                self.logger.info(
                    f'Label map saved with {num_labels} labels (source={label_map_source})'
                )

            # Save Stage 1 intermediates
            if self.save_intermediates:
                self._save_intermediate('step1_cleaned_mask', stage1_result['cleaned_mask'], is_mask=True)
                self._save_intermediate('step1_distance_map', stage1_result['distance_map'], is_mask=False)

            # Stage 2: Extremal points
            self.logger.info('Running Stage 2: Extremal points detection')
            min_distance_value = float(stage_kwargs.get('min_distance_value', 2.0))
            if scale > 1:
                # Keep physical threshold consistent after downsampling.
                min_distance_value = min_distance_value / float(scale)
                self.logger.info(
                    f'Adjusted min_distance_value to {min_distance_value:.3f} '
                    f'for downsample factor {scale}'
                )
            stage2_result = detect_extremal_points(
                stage1_result['cleaned_mask'],
                min_distance_value=min_distance_value,
                distance_map=stage1_result['distance_map'],
                edt_backend=effective_edt_backend,
                gpu_backend=effective_gpu_backend,
                retry_if_empty=True,
                skeleton_fallback=True,
                allow_cpu_edt=allow_cpu_edt,
            )
            self.results['stage2'] = stage2_result
            self.logger.info(
                f'Stage 2 complete: {len(stage2_result["extremal_points"])} extremal points'
            )

            # Save Stage 2 intermediates (extremal points as mask)
            if self.save_intermediates:
                ep_mask = np.zeros_like(stage1_result['cleaned_mask'], dtype=np.uint8)
                for ep in stage2_result['extremal_points']:
                    pos = ep['position']
                    idx = tuple(np.clip(np.round(pos).astype(int), 0, np.array(ep_mask.shape) - 1))
                    ep_mask[idx] = 1
                # Dilate to make visible
                ep_mask = binary_dilation(ep_mask, iterations=2).astype(np.uint8)
                self._save_intermediate('step2_extremal_points', ep_mask, is_mask=True)

            # Stage 4: Eikonal path tracing
            self.logger.info('Running Stage 4: Eikonal path tracing')
            stage4_result = extract_centerlines_via_eikonal(
                distance_map=stage1_result['distance_map'],
                extremal_points=stage2_result['extremal_points'],
                step_size=stage_kwargs.get('step_size', 0.1),
                max_iterations=stage_kwargs.get('max_iterations', 5000),
                gpu_backend=self.device if self.device in ('cuda', 'mps') else None,
                k_nearest=stage_kwargs.get('k_nearest'),
                max_pair_distance=stage_kwargs.get('max_pair_distance'),
            )
            self.results['stage4'] = stage4_result
            self.logger.info(
                f'Stage 4 complete: {len(stage4_result["centerlines"])} centerlines'
            )

            # Save Stage 4 intermediates (centerlines as mask)
            if self.save_intermediates:
                cl_mask = np.zeros_like(stage1_result['cleaned_mask'], dtype=np.uint8)
                bounds = np.array(cl_mask.shape) - 1
                for seg_data in stage4_result['centerlines'].values():
                    path = np.array(seg_data['path'])
                    for pt in path:
                        idx = tuple(np.clip(np.round(pt).astype(int), 0, bounds))
                        cl_mask[idx] = 1
                self._save_intermediate('step3_centerlines', cl_mask, is_mask=True)

            # Stage 5: Radius computation
            self.logger.info('Running Stage 5: Radius computation')
            stage5_result = compute_radii(
                centerlines=stage4_result['centerlines'],
                distance_map=stage1_result['distance_map'],
            )

            # Rescale centerlines back to original voxel space if cropped/downsampled
            # Note: stage5 shares reference to stage4 centerlines, so we deep copy before rescaling
            if scale != 1 or offset != (0, 0, 0):
                stage4_centerlines = copy.deepcopy(stage4_result['centerlines'])
                stage5_centerlines = copy.deepcopy(stage5_result['centerlines'])
                stage4_result['centerlines'] = self._rescale_centerlines(
                    stage4_centerlines, offset, scale
                )
                stage5_result['centerlines'] = self._rescale_centerlines(
                    stage5_centerlines, offset, scale
                )

            self.results['stage5'] = stage5_result
            self.logger.info(f'Stage 5 complete: radii assigned to all centerlines')

            # Save Stage 5 intermediates (radius-weighted centerlines)
            if self.save_intermediates:
                # Use original mask shape for output
                orig_img = nib.load(self.nifti_path)
                radius_vol = np.zeros(orig_img.shape, dtype=np.float32)
                bounds = np.array(radius_vol.shape) - 1
                for seg_data in stage5_result['centerlines'].values():
                    path = np.array(seg_data['path'])
                    radii = seg_data.get('radii', [1.0] * len(path))
                    for pt, r in zip(path, radii):
                        idx = tuple(np.clip(np.round(pt).astype(int), 0, bounds))
                        radius_vol[idx] = max(radius_vol[idx], r)
                self._save_intermediate('step4_radii', radius_vol, is_mask=False)

            # Stage 6: Bifurcation detection
            self.logger.info('Running Stage 6: Bifurcation detection')
            stage6_result = detect_bifurcations(
                stage5_result['centerlines'],
                contact_distance_threshold=stage_kwargs.get(
                    'contact_distance_threshold', 1.0
                ),
            )
            self.results['stage6'] = stage6_result
            self.logger.info(
                f'Stage 6 complete: {stage6_result["num_bifurcations"]} bifurcations detected'
            )

            # Save Stage 6 intermediates (bifurcation points)
            if self.save_intermediates:
                orig_img = nib.load(self.nifti_path)
                bif_mask = np.zeros(orig_img.shape, dtype=np.uint8)
                bounds = np.array(bif_mask.shape) - 1
                for bif in stage6_result.get('bifurcations', []):
                    pos = bif.get('position') or bif.get('point')
                    if pos is not None:
                        idx = tuple(np.clip(np.round(pos).astype(int), 0, bounds))
                        bif_mask[idx] = 1
                bif_mask = binary_dilation(bif_mask, iterations=3).astype(np.uint8)
                self._save_intermediate('step5_bifurcations', bif_mask, is_mask=True)

            # Stage 7: Graph construction
            self.logger.info('Running Stage 7: Graph construction')
            stage7_result = build_centerline_graph(
                stage5_result['centerlines'],
                stage6_result,
            )
            self.results['stage7'] = stage7_result
            self.logger.info(
                f'Stage 7 complete: {len(stage7_result["node_data"])} nodes, '
                f'{len(stage7_result["edge_data"])} edges'
            )

            # Export graph
            self.logger.info('Exporting centerline graph')
            export_result = export_graph(
                stage7_result,
                self.output_dir,
                basename='centerline',
            )
            self.results['export'] = export_result
            self.logger.info(
                f'Graph exported to {export_result["pickle_path"]}'
            )

            self.logger.info('Pipeline completed successfully')
            return self.results

        except Exception as e:
            self.logger.error(f'Pipeline failed: {e}', exc_info=True)
            raise

    def summary(self) -> dict:
        """Generate summary statistics from pipeline results.

        Returns
        -------
        dict
            Summary with num_centerlines, total_path_length, etc.
        """
        summary = {
            'num_components': self.results.get('stage1', {}).get(
                'num_components', 0
            ),
            'num_extremal_points': len(
                self.results.get('stage2', {}).get('extremal_points', [])
            ),
            'num_centerlines': len(
                self.results.get('stage4', {}).get('centerlines', {})
            ),
            'num_bifurcations': self.results.get('stage6', {}).get(
                'num_bifurcations', 0
            ),
            'num_nodes': len(
                self.results.get('stage7', {}).get('node_data', [])
            ),
            'num_edges': len(
                self.results.get('stage7', {}).get('edge_data', [])
            ),
        }

        # Compute total path length
        total_length = 0.0
        for centerline in (
            self.results.get('stage4', {}).get('centerlines', {}).values()
        ):
            path = np.array(centerline['path'])
            if len(path) > 1:
                diffs = np.diff(path, axis=0)
                lengths = np.linalg.norm(diffs, axis=1)
                total_length += float(lengths.sum())
        summary['total_centerline_length_mm'] = total_length

        return summary
