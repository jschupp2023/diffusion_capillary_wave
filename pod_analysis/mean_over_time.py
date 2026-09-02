import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

pod_file = Path("pod_analysis/pod_2d_r4000_0p35.h5")
output_file = pod_file.with_name(f"{pod_file.stem}_mean_over_time.png")

with h5py.File(pod_file, "r") as f:
    time = f["grid/time"][:]
    spatial_mean = f["preprocessing/frame_spatial_mean"][:]

plt.plot(time, spatial_mean, linewidth=0.8)
plt.xlabel("Time [s]")
plt.ylabel("Spatial mean [microns]")
plt.title("Spatial mean over time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to {output_file.resolve()}")
