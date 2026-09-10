import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from run_raw_capillary_energy_batch import discover, parse_args, run


def write_raw(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        x, y = np.linspace(0, 300, 7), np.linspace(0, 200, 5)
        xx, yy = np.meshgrid(x, y)
        for i in range(11):
            f.create_dataset(f'main/{i:05d}', data=np.asarray(.1*xx + .01*i*yy, dtype=np.float32))
        for key, array in [('x', x), ('y', y), ('t', np.linspace(0, .1, 11)), ('frames', np.arange(11))]:
            f.create_dataset('meta/'+key, data=array)
        for key in ['x', 'y', 'z']:
            f['meta'].attrs[key+'_units'] = 'microns'
        f['meta'].attrs['t_units'] = 'seconds'


class RawBatchTests(unittest.TestCase):
    def test_compact_collection_resume_errors_and_settings_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, out = root/'raw', root/'results'
            write_raw(raw/'0p000/Ca_ac_a_rep1/experiment.hdf5')
            write_raw(raw/'0p35/Ca_ac_b_rep2/experiment.hdf5')
            args = parse_args(['--raw-root', str(raw), '--output-dir', str(out), '--frames', '5'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            summaries = list(csv.DictReader((out/'recording_summary.csv').read_text().splitlines()))
            self.assertEqual({r['power'] for r in summaries}, {'0p000', '0p35'})
            self.assertTrue(all(r['temporal_mean_removed']=='False' for r in summaries))
            self.assertEqual(len(list(out.glob('*.h5'))), 1)
            self.assertFalse(list(out.glob('*.png')))
            with h5py.File(out/'raw_capillary_energy.h5') as f:
                for row in summaries:
                    g = f[row['recording_key']]
                    self.assertTrue(g.attrs['complete'])
                    self.assertEqual(len(g['frame_indices']), 5)
                    self.assertNotIn('modes', g)
                    np.testing.assert_allclose(g['E_exact'][:]@g['mean_weights'][:], float(row['Ebar_exact_J']), rtol=1e-13)
            with patch('run_raw_capillary_energy_batch.compute', side_effect=AssertionError('cached jobs recomputed')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            # A malformed file and ambiguous directory must not erase valid results.
            bad = raw/'0p35/Ca_ac_c_rep3/bad.h5'
            bad.parent.mkdir()
            with h5py.File(bad, 'w'):
                pass
            ambiguous = raw/'0p35/Ca_ac_d_rep4'
            ambiguous.mkdir()
            (ambiguous/'a.h5').touch(); (ambiguous/'b.hdf5').touch()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 1)
            self.assertEqual(len(list(csv.DictReader((out/'recording_summary.csv').read_text().splitlines()))), 2)
            self.assertIn('found 2', (out/'discovery_issues.csv').read_text())
            args.frames = 6
            with self.assertRaisesRegex(ValueError, 'different analysis settings'):
                run(args)
            self.assertEqual(json.loads((out/'configuration.json').read_text())['frames'], 5)

    def test_dry_run_has_no_output_and_power_filter_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root/'raw/0p04/Ca_ac_a_rep1/raw.h5')
            write_raw(root/'raw/0p20/Ca_ac_b_rep1/raw.h5')
            out = root/'out'
            args = parse_args(['--raw-root', str(root/'raw'), '--output-dir', str(out), '--powers', '0p20', '--dry-run'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            self.assertFalse(out.exists())
            jobs, issues = discover(root/'raw', ['0p20'])
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['power'], '0p20')
            self.assertEqual(issues, [])


if __name__ == '__main__':
    unittest.main()
