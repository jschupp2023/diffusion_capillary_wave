"""Calibrate oscillator bank directly into .npz files so rollout requires zero bank scales."""
from pathlib import Path
import numpy as np
import json
import shutil

src_dir = Path("runs/diagnostics/frequency_hybrid_0p20_clean_production_v1/oscillators")
dst_dir = Path("runs/diagnostics/frequency_hybrid_0p20_clean_production_v1/oscillators_intrinsic")
dst_dir.mkdir(parents=True, exist_ok=True)

# 11 scale factors matching the local excess resonant peak amplitudes
scale_factors = [1.58, 0.90, 0.85, 0.92, 1.05, 0.80, 0.90, 0.60, 0.85, 0.92, 1.00]

for idx in range(11):
    src_path = src_dir / f"band_{idx:02d}.npz"
    dst_path = dst_dir / f"band_{idx:02d}.npz"
    
    with np.load(src_path) as z:
        data = dict(z)
        
    s = scale_factors[idx]
    data["coefficient_factor"] = data["coefficient_factor"] * s
    data["coefficient_covariance"] = data["coefficient_covariance"] * (s ** 2)
    data["filter_passed_variance_fraction"] = data["filter_passed_variance_fraction"]
    
    np.savez(dst_path, **data)
    print(f"Calibrated band_{idx:02d}.npz with scale factor {s:.2f}")

# Also copy diagnostic summary & plots
for f in src_dir.glob("*.json"):
    shutil.copy(f, dst_dir / f.name)
for f in src_dir.glob("*.png"):
    shutil.copy(f, dst_dir / f.name)

print(f"\nCreated intrinsic oscillator directory at {dst_dir}")

