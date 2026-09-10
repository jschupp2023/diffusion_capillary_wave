import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

from raw_capillary_energy import run
from pod_capillary_energy import select_energy_frames


class RawCapillaryEnergyTests(unittest.TestCase):
    def test_sampled_raw_and_mean_removed_affine_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, pod = root/'raw.h5', root/'means.h5'
            x, y, t = np.linspace(0, 300, 7), np.linspace(0, 200, 5), np.array([0., .1, .3, .6, 1.])
            xx, yy = np.meshgrid(x, y)
            fields = np.array([.2*xx + c*yy for c in np.arange(5)*.1], dtype=np.float32)
            with h5py.File(raw, 'w') as f:
                main, meta = f.create_group('main'), f.create_group('meta')
                for i, field in enumerate(fields):
                    main.create_dataset(f'{i:05d}', data=field)
                for key, value in [('x', x), ('y', y), ('t', t), ('frames', np.arange(5))]:
                    meta.create_dataset(key, data=value)
                for key in ['x', 'y', 'z']:
                    meta.attrs[f'{key}_units'] = 'microns'
                meta.attrs['t_units'] = 'seconds'
            spatial = fields.mean(axis=(1, 2), dtype=float)
            centered = fields-spatial.astype(np.float32)[:, None, None]
            with h5py.File(pod, 'w') as f:
                for key, value in dict(source_file=str(raw), source_start_position=0, source_stop_position_exclusive=5,
                                       temporal_mean_field_removed=True, instantaneous_spatial_mean_removed=True).items():
                    f.attrs[key] = value
                for key, value in [('x', x), ('y', y), ('time', t)]:
                    f.create_dataset('grid/'+key, data=value)
                f.create_dataset('preprocessing/temporal_mean_field', data=centered.mean(axis=0)).attrs['units'] = 'microns'
                f.create_dataset('preprocessing/frame_spatial_mean', data=spatial)
            args = Namespace(input=raw, mean_from_pod=pod, frames=3, frame_seed=12345, gamma=.0728,
                             batch_size=2, output_dir=root/'out')
            with contextlib.redirect_stdout(io.StringIO()):
                rows = run(args)
            indices, weights = select_energy_frames(t, 3, 12345)
            slope = indices*.1
            area = 300e-6*200e-6
            for i, s in enumerate([.2**2+slope**2, (slope-.2)**2]):
                quad = .5*.0728*area*s
                exact = .0728*area*s/(np.sqrt(1+s)+1)
                np.testing.assert_allclose(rows[i]['Ebar_quad_J'], quad@weights, rtol=2e-6)
                np.testing.assert_allclose(rows[i]['Ebar_exact_J'], exact@weights, rtol=2e-6)
            with h5py.File(pod, 'a') as f:
                f.attrs['source_file'] = str(root/'wrong.h5')
            with self.assertRaisesRegex(ValueError, 'exact raw input'):
                run(args)


if __name__ == '__main__':
    unittest.main()
