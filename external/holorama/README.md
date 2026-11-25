# HolOrama whole-heart CT model (local only, not in git)

Used by `subprojects/heart_segmentation/` (nnU-Net 3d_fullres exported to ONNX; 14 cardiac structures).

- Source: https://github.com/AI-in-Cardiovascular-Medicine/HolOrama (MIT License), files
  `src/segmentation/weights/ct_seg/{ct_seg_u2net_small.onnx, ct_seg_u2net_small_fp16.onnx, config.json}` + `LICENSE`
- Pinned commit: `e6ce0460249be20df058cd204df332f9e01aa594` (2026-11-20, HolOrama 0.21.0)
- Papers: Mohammadi Kazaj P et al., arXiv:2607.11287 (2026); Stark AW et al., Comput Methods Programs Biomed 2025;109065

```bash
python subprojects/heart_segmentation/fetch_holorama_weights.py          # download, sha256-verified (~21 MB)
python subprojects/heart_segmentation/fetch_holorama_weights.py --check  # verify
```

`SOURCE.json` is written by the fetch script and records commit, checksums and fetch time.
Only this README is tracked; everything else in this folder is git-ignored.
