# NV-Reason-CT second reader (office / DGX Spark only)

NV-Reason-CT (NVIDIA, Sept 2026; arXiv 2609.27511; https://huggingface.co/nvidia/NV-Reason-CT) is an open 3D CT
vision-language model (3D ViT + Qwen 3.5 4B, 4.7B params) for chest and abdomen CT: structured reports, VQA,
chain-of-thought. Licence OpenMDW-1.1, research only. **CUDA only** (tested on H100 / L40S) and 10.6 GB of
weights, so it is not run on the Mac; this folder prepares inputs here and runs the model on the DGX Spark.

## Workflow

1. Mac: TotalSegmentator `total` with the lung lobes (already part of the cancer-screen chain), then
   `prepare_chest_crop.py` -> `<case>_chest.nii.gz` (thorax bbox + 25 mm, HU int16). The model auto-crops a
   192 mm cube at 2 mm; a vertex-to-abdomen CTA confuses its chest heuristic, hence the pre-crop.
2. Copy the crops to the Spark (~170 MB each at native 0.45 x 0.25 mm; the model resamples to 2 mm anyway, so use `--downsample-mm 1.0` (~15 MB) if bandwidth matters), not the full CTAs.
3. Spark, one-off env (transformers >= 5.6 conflicts with the VISTA env, keep it separate):
   ```bash
   git clone https://github.com/NVIDIA-Medtech/NV-Reason-CT && cd NV-Reason-CT
   conda create -n nvreason python=3.11 -y && conda activate nvreason
   pip install torch --index-url https://download.pytorch.org/whl/cu130   # GB10 / Blackwell
   uv pip install -r requirements.txt   # or pip
   python inference.py examples/example_1.nii.gz --region chest --prompt "write a structured chest CT report"  # smoke test
   ```
4. Spark: `python run_nv_reason_ct_batch.py --input-dir <crops> --out-dir <reports>` -> `<case>_nvreason.json`
   (structured report + five targeted screening questions: nodules, lymphadenopathy, thyroid, other masses, malignancy).
5. Mac: `parse_nv_reason_reports.py` -> CSV of keyword flags to join with `cancer_screen` features by case_id;
   the reader reads the report text, the flags only order the queue.

## Caveats
- Trained on chest/abdomen CT (CT-RATE etc.); no published validation on contrast CTA or non-gated stroke scans.
- 2 mm resampling: sub-6 mm nodules will mostly be invisible to it; TotalSegmentator `lung_nodules` + MONAI
  RetinaNet remain the nodule detectors, NV-Reason-CT is the report-level second opinion.
- Thyroid may fall outside the 192 mm chest cube in tall patients; check `extent_mm` in the crop JSON.
- Keep every output as research text; nothing here is a diagnosis.
