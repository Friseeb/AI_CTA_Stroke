# STOPSTORM cardiac substructure models (local only, not in git)

Used by `subprojects/valve_calcium/` for the learned mitral valve label.

- Source: van der Pol et al., *Radiother Oncol* 2025, doi 10.1016/j.radonc.2024.110610
- Record: https://zenodo.org/records/13752275 (CC BY 4.0, research use only)
- File: `model603.zip` (Network B, "small structures": 4 valves + 4 coronary segments), ~232 MB

```bash
cd external/stopstorm
curl -L -o model603.zip "https://zenodo.org/api/records/13752275/files/model603.zip/content"
mkdir -p nnUNet_results && unzip -q model603.zip -d nnUNet_results
# -> nnUNet_results/Dataset603_STST_small_full/nnUNetTrainer__nnUNetPlans__3d_fullres/
```

Inference needs nnU-Net v2 (`nnUNetv2_predict`, present in the `totalseg-mac` conda env);
the driver sets `nnUNet_results` to this folder automatically.
