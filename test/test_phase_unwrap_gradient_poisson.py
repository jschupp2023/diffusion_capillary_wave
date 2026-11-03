"""Unit tests for gradient-domain Poisson phase unwrap restoration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
    restore_window,
)

HDF5_DATA_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")


def test_clean_frame_untouched():
    """Verify that a smooth physical frame with no jumps is left 100% untouched."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    # Smooth wave with max gradient ~ 0.5 um
    clean_wave = (0.5 * np.sin(X) * np.cos(Y)).astype(np.float32)

    res = restore_frame_poisson(clean_wave, q=DEFAULT_Q, tau=DEFAULT_TAU)
    assert not res.was_modified
    assert res.orig_jump_count == 0
    assert np.array_equal(res.restored, clean_wave)
    assert np.all(res.cycles_inferred == 0)


def test_synthetic_patch_step_restored():
    """Verify that an artificial integer-cycle plateau is accurately flattened."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    clean_wave = (0.5 * np.sin(X) * np.cos(Y)).astype(np.float64)

    corrupted = clean_wave.copy()
    # Add a +3*q plateau to a central square
    q = DEFAULT_Q
    corrupted[30:70, 30:70] += 3.0 * q

    res = restore_frame_poisson(corrupted, q=q, tau=DEFAULT_TAU)
    assert res.was_modified
    assert res.orig_max_dx > 5.0
    assert res.rest_max_dx < 1.0
    assert np.allclose(res.restored, clean_wave, atol=0.2)


def test_real_hdf5_discontinuous_frame_1553():
    """Test restoration on real experimental frame 1553 with a 75 um jump."""
    if not HDF5_DATA_PATH.exists():
        return
    with h5py.File(HDF5_DATA_PATH, "r") as h5:
        frame_1553 = np.asarray(h5["main"]["01553"], dtype=np.float32)

    # Integer cycle restoration
    res_int = restore_frame_poisson(frame_1553, q=DEFAULT_Q, tau=DEFAULT_TAU, method="integer")
    assert res_int.was_modified
    assert res_int.orig_max_dx > 70.0
    assert res_int.rest_max_dx < 5.0
    assert res_int.n_pixels_modified > 0

    # Poisson continuous restoration
    res_poiss = restore_frame_poisson(frame_1553, q=DEFAULT_Q, tau=DEFAULT_TAU, method="poisson")
    assert res_poiss.was_modified
    assert res_poiss.rest_max_dx < 5.0


def test_real_hdf5_clean_frame_2300():
    """Verify that frame 2300 (which was erroneously altered by tau=1.8) is left 100% untouched."""
    if not HDF5_DATA_PATH.exists():
        return
    with h5py.File(HDF5_DATA_PATH, "r") as h5:
        frame_2300 = np.asarray(h5["main"]["02300"], dtype=np.float32)

    res = restore_frame_poisson(frame_2300, q=DEFAULT_Q, tau=DEFAULT_TAU, method="integer")
    assert not res.was_modified
    assert np.array_equal(res.restored, frame_2300)
    assert res.n_pixels_modified == 0


def test_real_hdf5_clean_frame_20979():
    """Verify that a naturally clean frame from the dataset is not modified."""
    if not HDF5_DATA_PATH.exists():
        return
    with h5py.File(HDF5_DATA_PATH, "r") as h5:
        frame_clean = np.asarray(h5["main"]["20979"], dtype=np.float32)

    res = restore_frame_poisson(frame_clean, q=DEFAULT_Q, tau=DEFAULT_TAU)
    assert not res.was_modified
    assert np.array_equal(res.restored, frame_clean)


if __name__ == "__main__":
    print("Running test_clean_frame_untouched...")
    test_clean_frame_untouched()
    print("PASS")

    print("Running test_synthetic_patch_step_restored...")
    test_synthetic_patch_step_restored()
    print("PASS")

    print("Running test_real_hdf5_discontinuous_frame_1553...")
    test_real_hdf5_discontinuous_frame_1553()
    print("PASS")

    print("Running test_real_hdf5_clean_frame_2300...")
    test_real_hdf5_clean_frame_2300()
    print("PASS")

    print("Running test_real_hdf5_clean_frame_20979...")
    test_real_hdf5_clean_frame_20979()
    print("PASS")
    print("All unit tests passed successfully!")

    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

