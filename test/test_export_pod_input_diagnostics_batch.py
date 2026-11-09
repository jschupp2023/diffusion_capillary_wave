import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.export_pod_input_diagnostics_batch import parse_args, run


class ExportPodInputDiagnosticsBatchTests(unittest.TestCase):
    def make_case(self, root: Path) -> tuple[Path, Path]:
        source_dir = root / "sources" / "0p07" / "rep1"
        reduced_dir = root / "reduced" / "0p07" / "rep1"
        source_dir.mkdir(parents=True)
        reduced_dir.mkdir(parents=True)
        source = source_dir / "recording_cleaned_hp50hz.hdf5"
        pod = reduced_dir / "pod_2d_r2.h5"

        n_frames = 64
        sampling_hz = 1000.0
        time = np.arange(n_frames, dtype=float) / sampling_hz
        x = np.linspace(0.0, 50.0, 6)
        y = np.linspace(0.0, 40.0, 5)
        xx, yy = np.meshgrid(x, y)
        frame_numbers = np.arange(100, 100 + n_frames)
        fields = np.asarray(
            [
                0.02 * xx
                + 0.01 * yy
                + np.sin(2 * np.pi * 125.0 * instant) * (0.03 * xx - 0.01 * yy)
                for instant in time
            ],
            dtype=np.float32,
        )
        mean_field = fields.mean(axis=0, dtype=np.float64)

        with h5py.File(source, "w") as output:
            output.attrs.update(
                highpass_filter_applied=True,
                highpass_cutoff_hz=50.0,
                highpass_filter_order=4,
                highpass_filter_type="butterworth_zero_phase",
            )
            meta = output.create_group("meta")
            meta.attrs.update(
                t_units="seconds",
                x_units="microns",
                y_units="microns",
                z_units="microns",
            )
            meta.create_dataset("t", data=time)
            meta.create_dataset("x", data=x)
            meta.create_dataset("y", data=y)
            meta.create_dataset("frames", data=frame_numbers)
            main = output.create_group("main")
            for number, field in zip(frame_numbers, fields, strict=True):
                main.create_dataset(f"{number:05d}", data=field)

        with h5py.File(pod, "w") as output:
            output.attrs.update(
                rank=2,
                source_file=str(source.resolve()),
                source_start_position=0,
                source_stop_position_exclusive=n_frames,
                temporal_mean_field_removed=True,
                instantaneous_spatial_mean_removed=False,
            )
            output.create_dataset("pod/modes", data=np.zeros((2, len(y), len(x))))
            output.create_dataset("grid/time", data=time)
            output.create_dataset("grid/x", data=x)
            output.create_dataset("grid/y", data=y)
            mean = output.create_dataset(
                "preprocessing/temporal_mean_field", data=mean_field
            )
            mean.attrs.update(units="microns", subtracted_before_pod=True)
        return source, pod

    def test_compact_export_accepts_rep_folder_and_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_case(root)
            output_dir = root / "export"
            args = parse_args(
                [
                    "--reduced-root",
                    str(root / "reduced"),
                    "--raw-root",
                    str(root / "sources"),
                    "--output-dir",
                    str(output_dir),
                    "--scratch-root",
                    str(root / "scratch"),
                    "--pod-rank",
                    "2",
                    "--nperseg",
                    "16",
                    "--energy-frames",
                    "16",
                    "--energy-batch-size",
                    "4",
                ]
            )
            self.assertEqual(run(args), 0)
            archive = output_dir / "pod_input_diagnostics.h5"
            with h5py.File(archive, "r") as result:
                group = result["recordings/0p07/rep1"]
                self.assertTrue(group.attrs["complete"])
                self.assertEqual(group["center_psd/frequency_hz"].shape, (9,))
                self.assertEqual(group["center_psd/power_spectral_density"].shape, (9,))
                self.assertEqual(group["capillary_energy/E_exact_J"].shape, (16,))
                self.assertGreater(
                    group["capillary_energy"].attrs["Ebar_exact_J"], 0.0
                )
                self.assertGreaterEqual(
                    group["capillary_energy"].attrs["Ebar_quadratic_J"],
                    group["capillary_energy"].attrs["Ebar_exact_J"],
                )
            self.assertEqual(run(args), 0)
            self.assertTrue((output_dir / "recording_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
