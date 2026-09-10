# Raw capillary-energy batch on the lab computer

The default raw-data location follows `run_pod_batch.py` and `run_center_psd_comparison_batch.py`:

```
/disk/hyk049/DHM_new_experiment/
  0p04/
    Ca_ac_..._rep1/
      <experiment>.hdf5
  0p35/
    Ca_ac_..._rep1/
      <experiment>.hdf5
```

One direct .h5/.hdf5 file is required per Ca_ac recording directory. All power folders are included by default, including zero and low input. Ambiguous inputs and failures are recorded, not silently discarded. Power labels follow folder names; raw filenames are retained separately.

## Code and environment

Copy `raw_capillary_energy_lab_bundle.zip` to the lab computer and extract it into a new code directory. The bundle contains this runner, its local Python dependencies, tests and this guide. No experimental data are bundled. Use the lab's existing Python environment with NumPy, h5py, SciPy and Matplotlib installed. The scripts use NumPy 2's trapezoid API in tests and some optional utilities; the laptop validation used Python 3.12 and NumPy 2.

From the extracted directory, inspect discovery without reading height frames or creating results:

```bash
python run_raw_capillary_energy_batch.py \
  --output-dir /disk/hyk049/jschupp/raw_capillary_energy_2000frames \
  --dry-run
```

Optional two-recording trial:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python run_raw_capillary_energy_batch.py \
  --output-dir /disk/hyk049/jschupp/raw_capillary_energy_2000frames \
  --frames 2000 --frame-seed 12345 --gamma 0.0728 --max-files 2
```

Then process all recordings (the trial's completed files are reused):

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python run_raw_capillary_energy_batch.py \
  --output-dir /disk/hyk049/jschupp/raw_capillary_energy_2000frames \
  --frames 2000 --frame-seed 12345 --gamma 0.0728
```

Use `--raw-root` to override the raw-data location, or `--powers 0p20 0p35` to restrict the input powers. The output folder is created automatically. Normal reruns resume; `--overwrite` recomputes jobs. Changes to sample count, seed or gamma require a new output folder to prevent mixed settings. Avoid concurrent writers to one output folder.

## What is computed and transferred

The main result is the sampled time mean of exact geometric capillary excess energy, in joules. The inexpensive quadratic result is saved alongside it. There is no temporal/spatial mean subtraction, POD projection, smoothing, or amplitude normalization. The calibrated raw heights include the static surface and any measurement noise. Heights and spatial axes are converted to metres.

The same derivative and spatial quadrature functions used in the validated single-recording analysis are reused. Each recording gets 2,000 distinct uniformly random frames, seed 12345. Full-grid trapezoidal time weights are divided by each frame's inclusion probability; the resulting time mean is a sampling estimate. The same seed gives the same indices for equal-length recordings. Sampling uncertainty is distinct from experimental variability across repetitions.

Transfer **the entire results folder** back to the laptop:

- `recording_summary.csv`: exact/quadratic means and diagnostic/provenance fields for every completed recording.
- `power_summary.csv`: mean, median and SD across recording-level means, grouped by power.
- `raw_capillary_energy.h5`: sampled per-frame exact/quadratic energies, indices, timestamps, integration weights, x/y grids, units and source metadata.
- `configuration.json`, `manifest.csv`, `discovery_issues.csv`, `batch_*.log`, `README.md`.

The archive never contains raw height fields. Six 2,000-element vectors are about 96 kB per recording before compression; grids and metadata add a little more. For roughly 160 recordings, expect an archive on the order of tens of MB rather than GB. The exact size depends on the number of recordings and compression.

Optional packaging for transfer:

```bash
tar -czf raw_capillary_energy_2000frames.tar.gz \
  -C /disk/hyk049/jschupp raw_capillary_energy_2000frames
```

The CSVs and HDF5 archive can be read on the laptop without access to the lab's original raw-data paths. Read the failure/discovery logs to establish which recordings are included. The runner returns a nonzero exit status if any computation or discovery issue occurred, while retaining completed results.

## Local validation

Tiny synthetic raw surfaces test the numeric calculation, matched mean preprocessing for the single-file tool, inclusion of low powers, output structure, repeated-run caching, invalid input isolation, configuration mismatch rejection and discovery-only operation. No full lab batch was run on the laptop.

```bash
python -m unittest test_raw_capillary_energy.py test_raw_capillary_energy_batch.py
```
