"""Shared capillary-energy numerics extracted unchanged from pod_capillary_energy."""
import numpy as np


def length_scale(unit: str) -> float:
    unit = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {"m": 1., "meter": 1., "meters": 1., "metres": 1., "metre": 1.,
              "mm": 1e-3, "millimeters": 1e-3, "um": 1e-6, "micron": 1e-6,
              "microns": 1e-6, "micrometers": 1e-6, "micrometres": 1e-6,
              "nm": 1e-9, "nanometers": 1e-9}
    if unit not in scales:
        raise ValueError(f"Unsupported spatial/displacement unit {unit!r}; cannot compute physical energy.")
    return scales[unit]


def quadrature_weights(grid: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature, including half weights at the boundaries."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Spatial grids must be finite, strictly increasing and have >=3 points.")
    steps = np.diff(grid)
    return np.r_[steps[0]/2, (steps[:-1]+steps[1:])/2, steps[-1]/2]


def capillary_stiffness(modes: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                        geometry: str = "surface", gamma: float = 1.) -> tuple[np.ndarray, dict]:
    """Build gamma times the discrete gradient Gram matrix of dimensionless modes.

    Central finite differences inside; second-order one-sided derivatives at
    boundaries; trapezoidal quadrature. No periodic boundary is assumed.
    """
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 3 or modes.shape[1:] != (len(y_m), len(x_m)) or not np.isfinite(modes).all():
        raise ValueError("Expected finite modes with shape (rank, y, x).")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Surface tension must be positive and finite.")
    wx, wy = quadrature_weights(x_m), quadrature_weights(y_m)
    center = int(np.argmin(np.abs(y_m - .5*(y_m[0]+y_m[-1]))))
    if geometry == "centerline":
        gx = np.gradient(modes[:, center, :], x_m, axis=-1, edge_order=2)
        matrix = (gx*wx) @ gx.T
        measure = float(wx.sum())
    elif geometry == "surface":
        weight = np.outer(wy, wx).ravel()
        gx = np.gradient(modes, x_m, axis=-1, edge_order=2).reshape(len(modes), -1)
        matrix = (gx*weight) @ gx.T
        del gx
        gy = np.gradient(modes, y_m, axis=-2, edge_order=2).reshape(len(modes), -1)
        matrix += (gy*weight) @ gy.T
        measure = float(weight.sum())
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    matrix *= gamma
    symmetry_error = float(np.linalg.norm(matrix-matrix.T, ord="fro"))
    matrix = .5*(matrix+matrix.T)
    info = dict(measure=measure, centerline_y_index=center, centerline_y_m=float(y_m[center]),
                symmetry_error_before_symmetrizing=symmetry_error)
    return matrix, info


def matrix_diagnostics(matrix: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh(.5*(matrix+matrix.T))
    tolerance = 1e-10*max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if eig[0] < -tolerance:
        raise ValueError(f"Capillary stiffness is not positive semidefinite: minimum eigenvalue {eig[0]:.6g}")
    return dict(symmetry_error=float(np.linalg.norm(matrix-matrix.T, ord="fro")),
                minimum_eigenvalue=float(eig[0]), maximum_eigenvalue=float(eig[-1]),
                psd_tolerance=tolerance, positive_semidefinite=True)


def energy_series(coordinates_m: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = .5*np.einsum("ti,ij,tj->t", coordinates_m, matrix, coordinates_m, optimize=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite capillary energy.")
    tolerance = 1e-10*max(float(np.max(np.abs(result))), np.finfo(float).tiny)
    if np.min(result) < -tolerance:
        raise ValueError("Negative capillary energy beyond numerical tolerance.")
    return np.maximum(result, 0)


def energy_statistics(t: np.ndarray, energy: np.ndarray) -> dict:
    if len(t) != len(energy) or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Energy needs a matching, strictly increasing time grid.")
    return dict(time_averaged_energy=float(np.trapezoid(energy, t)/(t[-1]-t[0])),
                sample_mean=float(np.mean(energy)), median=float(np.median(energy)),
                std=float(np.std(energy, ddof=1)), minimum=float(np.min(energy)),
                q05=float(np.quantile(energy, .05)), q25=float(np.quantile(energy, .25)),
                q75=float(np.quantile(energy, .75)), q95=float(np.quantile(energy, .95)),
                maximum=float(np.max(energy)))


def field_energies(field_m, x_m, y_m, geometry='surface'):
    """Geometric quadratic/exact excess per frame with the stiffness discretization.

    s/(sqrt(1+s)+1) is algebraically sqrt(1+s)-1 without cancellation at small slopes.
    The physical energies are gamma times these geometric integrals.
    """
    wx, wy = quadrature_weights(x_m), quadrature_weights(y_m)
    if geometry == 'surface':
        gx = np.gradient(field_m, x_m, axis=-1, edge_order=2)
        gy = np.gradient(field_m, y_m, axis=-2, edge_order=2)
        s = gx**2 + gy**2
        weight = np.outer(wy, wx).ravel()
    elif geometry == 'centerline':
        center = int(np.argmin(np.abs(y_m - .5*(y_m[0]+y_m[-1]))))
        gx = np.gradient(field_m[:, center, :], x_m, axis=-1, edge_order=2)
        s = gx**2
        weight = wx
    else:
        raise ValueError(f'Unknown geometry: {geometry}')
    s = s.reshape(len(field_m), -1)
    quad = .5*(s @ weight)
    exact = (s/(np.sqrt(1+s)+1)) @ weight
    if not np.isfinite(quad).all() or not np.isfinite(exact).all():
        raise ValueError('Nonfinite reconstructed surface energy.')
    if np.any(exact < 0) or np.any(exact > quad*(1+1e-12)):
        raise ValueError('Expected 0 <= exact surface energy <= quadratic energy.')
    return quad, exact

