#!/usr/bin/env python3
"""Run NV-Reason-CT (nvidia/NV-Reason-CT, OpenMDW-1.1, research only) over thorax crops.  DGX Spark / CUDA only.

  python run_nv_reason_ct_batch.py --input-dir <dir with *_chest.nii.gz> --out-dir <dir> [--prompts default|screen]

Per case writes <case>_nvreason.json with the structured report and the answers to targeted screening questions.
Model is loaded once.  Based on the model card snippet (transformers >= 5.6, trust_remote_code).
"""
import argparse, json, sys, time
from pathlib import Path

REPORT_PROMPT = "Write a structured chest CT report. List findings by organ system, then an impression."
SCREEN_QUESTIONS = {
    "lung_nodules": "Are there any pulmonary nodules or masses? For each, give lobe, approximate size in mm, and whether solid, part-solid or ground-glass.",
    "lymphadenopathy": "Is there mediastinal, hilar or supraclavicular lymphadenopathy? Give station and short-axis size in mm for enlarged nodes.",
    "thyroid": "Describe the thyroid gland. Are there nodules or a mass? Give size in mm if present.",
    "masses_other": "Are there any other masses, suspicious lesions, pleural effusion, or bone lesions? Answer concisely.",
    "malignancy": "Overall, is there any finding suspicious for malignancy that would need follow-up? Answer yes or no, then justify in one sentence.",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*_chest.nii.gz"); ap.add_argument("--model", default="nvidia/NV-Reason-CT")
    ap.add_argument("--max-new-tokens", type=int, default=2048); ap.add_argument("--thinking", action="store_true", help="request structured thinking output if supported")
    ap.add_argument("--skip-questions", action="store_true")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available():
        raise SystemExit("NV-Reason-CT needs a CUDA GPU (DGX Spark); this host has none")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def ask(ct_path: Path, prompt: str) -> str:
        inputs = processor(text=prompt, images3d=[str(ct_path)], anatomy_region="chest", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens)
        return processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    for ct in sorted(a.input_dir.glob(a.glob)):
        case = ct.name.replace("_chest.nii.gz", "")
        out = a.out_dir / f"{case}_nvreason.json"
        if out.is_file():
            print(case, "skipped"); continue
        t1 = time.time()
        rec = {"case_id": case, "input": str(ct), "model": a.model, "report": ask(ct, REPORT_PROMPT), "answers": {}}
        if not a.skip_questions:
            for k, q in SCREEN_QUESTIONS.items():
                rec["answers"][k] = ask(ct, q)
        rec["seconds"] = round(time.time() - t1, 1)
        rec["intended_use"] = "research; model output for reader review, not a diagnosis (OpenMDW-1.1)"
        out.write_text(json.dumps(rec, indent=2)); print(case, f"{rec['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
