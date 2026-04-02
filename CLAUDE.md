# CLAUDE.md — AI_CTA_Stroke / SLAAO Project

Read this file before writing any code or command.

---

## 1. TERMINAL COMMANDS

**Never run batches autonomously.** Always write the command for Federica to paste.

All commands must be **single-line PowerShell** using `conda run` to embed the environment:
```powershell
conda run -n cardiac-ct-explorer python scripts/run_full_segmentation_batch.py --bids-root "C:\Users\spost\Desktop\CT_image\SLAAOBIDS" --limit 1
```

Do not use backslash continuation — PowerShell does not support it.
Do not write `conda activate` as a separate step.

---

## 2. PROGRESS BAR — ALWAYS REQUIRED

Every batch script must include a **tqdm** progress bar that updates **live during execution** (not at the end).

```python
from tqdm import tqdm

for subject in tqdm(subjects, desc="Segmentation", unit="case", dynamic_ncols=True):
    ...
```

- Use `dynamic_ncols=True` to ensure live updating
- Never buffer output or print only at the end
- Never use `--quiet-subprocess` — terminal output must stay visible

---

## 3. TIME-CONSUMING BATCHES — CHECKPOINT STRATEGY

A batch is **time-consuming** if it takes **> 30 min** or processes **> 20 subjects**.
These require checkpoint/resume logic before writing anything else.

### 3a. Skip-if-done
Check for the expected output file at the start of each iteration and skip if already present:
```python
if output_path.exists():
    log_row["status"] = "skipped"
    continue
```

### 3b. Per-subject CSV log
Write a `batch_log.csv` with: `subject_id`, `status` (done/failed/skipped), `timestamp`, `error_message`.

### 3c. Save intermediate outputs at each sub-stage
If a pipeline has distinct sub-stages (e.g. preprocessing → feature extraction), each stage must save its output to disk before the next stage begins. Never chain stages in memory. This way, if a later stage fails, the earlier stage does not need to be repeated.

### 3d. Always suggest --limit 1 first
Before any large batch, suggest a single-subject test run:
```powershell
conda run -n cardiac-ct-explorer python scripts/run_full_segmentation_batch.py --bids-root "..." --limit 1
```

---

## 4. BEFORE WRITING ANY SCRIPT — CHECKLIST

- [ ] tqdm with `dynamic_ncols=True` included?
- [ ] Command is a single PowerShell line with `conda run -n <env>`?
- [ ] Skip-if-done + CSV log present (if > 30 min or > 20 subjects)?
- [ ] Each sub-stage saves output to disk before the next begins?
- [ ] `--limit 1` test run suggested?
