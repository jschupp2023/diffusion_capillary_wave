"""Checks for pairing, aggregation, common scoring support and cache reuse."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from data_analysis.bispectrum.run_pod_bispectral_batch import (
    aggregate_errors, cache_path, compute_pod, paired_errors, pod_cache_key, select_entries,
)


def make_case(rep, level, offset):
    frequency = np.array([100., 200., 400.])
    raw = {"frequency_hz": frequency, "complex_bispectrum": np.full((3, 3), 10.**level, complex),
           "bicoherence": np.full((3, 3), .1 * rep)}
    return {"power": "0p35", "rep": rep, "experiment": f"test_rep{rep}", "raw": raw,
            "pod": {10: {"complex_bispectrum": raw["complex_bispectrum"] * 10.**offset,
                         "bicoherence": raw["bicoherence"] + .01 * rep},
                    100: {"complex_bispectrum": raw["complex_bispectrum"].copy(),
                          "bicoherence": raw["bicoherence"].copy()}}}


class BatchScoresTests(unittest.TestCase):
    def test_matching_repetitions_are_scored_before_averaging(self):
        cases = [make_case(1, 0, 1), make_case(2, 10, 3)]
        rows, coverage = paired_errors(cases, [10, 100], 500)
        self.assertEqual(len(rows), 2 * 2 * 2 * 3)
        for row in rows:
            expected = (row["rep"] * 2 - 1 if row["metric"] == "bispectrum_log"
                        else .01 * row["rep"] if row["metric"] == "bicoherence" else 0)
            self.assertAlmostEqual(row["error"], expected if row["rank"] == 10 else 0)
        groups = aggregate_errors(rows)
        item = next(g for g in groups if (g["rank"], g["metric"], g["domain"]) == (10, "bispectrum_log", "full"))
        self.assertEqual(item["n_repetitions"], 2)
        self.assertAlmostEqual(item["mean"], 2)
        self.assertAlmostEqual(item["std"], np.sqrt(2))
        self.assertAlmostEqual(item["q25"], 1.5)
        self.assertAlmostEqual(item["q75"], 2.5)
        self.assertEqual(coverage["full"]["retained_log_area_fraction"], 1)

    def test_all_cases_and_ranks_use_same_finite_mask(self):
        cases = [make_case(1, 0, 1), make_case(2, 2, 3)]
        cases[1]["pod"][100]["bicoherence"][0, 0] = np.nan
        rows, coverage = paired_errors(cases, [10, 100], 500)
        self.assertAlmostEqual(coverage["full"]["retained_log_area_fraction"], 15/16)
        for row in rows:
            self.assertEqual(row["valid_unique_frequency_pairs"], coverage[row["domain"]]["valid_unique_frequency_pairs"])
        cases[1]["pod"][100]["bicoherence"][:] = np.nan
        with self.assertRaisesRegex(ValueError, "No common finite"):
            paired_errors(cases, [10, 100], 500)

    def test_single_repetition_std_is_undefined(self):
        rows, _ = paired_errors([make_case(1, 0, 1)], [10], 500)
        self.assertTrue(all(g["std"] is None for g in aggregate_errors(rows)))

    def test_missing_repetition_is_rejected(self):
        manifest = [{"power": "0p20", "rep": 1}, {"power": "0p35", "rep": 1}]
        with self.assertRaisesRegex(ValueError, "Missing"):
            select_entries(manifest, ["all"], [1, 2])
        self.assertEqual(select_entries(manifest, ["0p35"], [1]), [manifest[1]])


class CacheTests(unittest.TestCase):
    def test_resume_avoids_reconstruction_and_rejects_wrong_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pod.npz"
            case = make_case(1, 0, 1)
            key = {"rank": 10}
            np.savez_compressed(path, frequency_hz=case["raw"]["frequency_hz"],
                                **case["pod"][10], cache_key_json=np.asarray(json.dumps(key)))
            with patch("run_pod_bispectral_batch.load_signals", side_effect=AssertionError("Must reuse")):
                result, status = compute_pod(case, 10, path, key, False)
                self.assertEqual(status, "cached")
                np.testing.assert_array_equal(result["bicoherence"], case["pod"][10]["bicoherence"])
                with self.assertRaisesRegex(ValueError, "key mismatch"):
                    compute_pod(case, 100, path, {"rank": 100}, False)

    def test_rank_and_changed_source_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pod, raw = root / "pod.h5", root / "raw.npz"
            pod.write_bytes(b"pod")
            raw.write_bytes(b"raw")
            case = {"power": "0p35", "experiment": "rep1", "pod_file": pod,
                    "raw_spectrum": raw, "config": {"wavelet": "cgau1"}}
            key = pod_cache_key(case, 10)
            initial = cache_path(root, case, 10, key)
            self.assertNotEqual(initial, cache_path(root, case, 100, pod_cache_key(case, 100)))
            pod.write_bytes(b"changed pod")
            self.assertNotEqual(initial, cache_path(root, case, 10, pod_cache_key(case, 10)))


if __name__ == "__main__":
    unittest.main()
