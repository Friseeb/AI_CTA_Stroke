import numpy as np
import nibabel as nib
from pathlib import Path

out_dir = Path(__file__).parents[0]
cta_path = out_dir / "cta_synth.nii.gz"

shape = (32, 32, 32)
data = np.zeros(shape, dtype=np.float32)
# Simple bright vessel: a vertical line
data[16, 16, :] = 300.0

affine = np.diag([1.0, 1.0, 1.0, 1.0])
nib.save(nib.Nifti1Image(data, affine), str(cta_path))
print(f"Wrote synthetic CTA to {cta_path}")
