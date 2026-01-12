# AI CTA Stroke Project

**Title:** Leveraging Artificial Intelligence for Improved Stroke Prediction and Prevention Using CT Angiography

**IRF 2025 Application**

## Project Structure

- `docs/` - Documentation, IRF materials, protocols, figures
- `data/` - Raw CTA scans (DICOM), BIDS-converted, secure clinical data
- `python/` - Analysis, preprocessing, AI models, visualization, CLI tools
- `r/` - Statistical analysis (Cox models, SGBA, calibration)
- `scripts/` - Shell runners for pipelines
- `logs/` - Pipeline execution logs
- `outputs/` - Models, predictions, figures, tables
- `results/` - Final results and reports
- `compliance/` - Ethics, SGBA, data governance
- `tmp/` - Temporary/scratch space

## Quick Start

1. Set up environment: `python -m venv env && source env/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Configure data paths in `config.yaml`
4. Run BIDS conversion: `bash scripts/run_bids_convert.sh`
5. Train model: `bash scripts/run_train_resnet.sh`
6. Run inference: `bash scripts/run_inference.sh`
7. Statistical analysis: `Rscript scripts/run_stats.R`

## Key Files

- IRF application: `docs/irf_2025/`
- Protocols: `docs/protocols/`
- Model training: `python/models/`
- Statistical analysis: `r/analysis/cox_models.R`
