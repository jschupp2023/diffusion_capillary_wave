import tempfile
import unittest
from pathlib import Path

import h5py
import matplotlib.image as mpimg
import numpy as np

from data_analysis.pod.raw_surface_snapshot import save_raw_snapshot


class RawSurfaceSnapshotTests(unittest.TestCase):
    def test_writes_selected_full_grid_frame_at_requested_pixel_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.h5"
            x = np.linspace(-2.0, 2.0, 7)
            y = np.linspace(-1.0, 1.0, 5)
            with h5py.File(source, "w") as h5:
                main = h5.create_group("main")
                meta = h5.create_group("meta")
                meta.create_dataset("t", data=[0.0, 0.25, 0.5])
                meta.create_dataset("x", data=x)
                meta.create_dataset("y", data=y)
                meta.create_dataset("frames", data=[10, 20, 30])
                meta.attrs["t_units"] = "seconds"
                meta.attrs["x_units"] = "microns"
                meta.attrs["y_units"] = "microns"
                meta.attrs["z_units"] = "microns"
                for frame_number in (10, 20, 30):
                    main.create_dataset(
                        f"{frame_number:05d}",
                        data=np.add.outer(y, x) + frame_number,
                    )

            output = root / "snapshot.png"
            returned = save_raw_snapshot(
                source, 1, output=output, dpi=50, width=4, height=3
            )

            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.is_file())
            image = mpimg.imread(output)
            self.assertEqual(image.shape[:2], (150, 200))

    def test_rejects_invalid_limits_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.h5"
            with h5py.File(source, "w") as h5:
                main = h5.create_group("main")
                meta = h5.create_group("meta")
                meta.create_dataset("t", data=[0.0])
                meta.create_dataset("x", data=[0.0, 1.0])
                meta.create_dataset("y", data=[0.0, 1.0])
                main.create_dataset("00000", data=np.zeros((2, 2)))
            output = root / "snapshot.png"
            output.touch()
            with self.assertRaises(FileExistsError):
                save_raw_snapshot(source, 0, output=output, dpi=10)
            with self.assertRaises(ValueError):
                save_raw_snapshot(
                    source,
                    0,
                    output=output,
                    dpi=10,
                    zmin=1.0,
                    zmax=0.0,
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
