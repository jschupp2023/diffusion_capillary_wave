"""Run the pre-unseen-data stochastic OpInf workflow from ``main.ipynb``.

All segmented realizations are used to compute the POD basis, reduced
experimental statistics, and stochastic reduced-order model. The script stops
before the notebook's unseen-data train/test split. Figures and numerical
results are written to one output directory.

Example
-------
    conda run --no-capture-output -n capillarywave python \
        run_main_analysis.py 0p15 --rank 19

    # Paper-author settings for the 0p35 case.
    conda run --no-capture-output -n capillarywave python \
        run_main_analysis.py 0p35 --rank 13 \
        --a-regularization 1e3 --b-regularization 0 \
        --n-regularization-min 1e5 --n-regularization-max 1e10 \
        --regularization-count 10 --h-regularization 1e4
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import helpers as sto_opinf
import PSD_utils


POWER_LABELS = {
    "0p04": list("abcdefghijklmnop"),
    "0p07": list("abcdefghijklnop"),
    "0p08": list("abcefghijklnop"),
    "0p10": list("bcdefhijklmnopq"),
    "0p15": list("abcdefghijklmnop"),
    "0p18": list("abcdefghijkmnop"),
    "0p20": list("abcdefghijklmnop"),
    "0p25": list("abcdefghijklmno"),
    "0p30": list("abcdefghijklmnop"),
    "0p35": list("abcdefhijklmnop"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power", choices=POWER_LABELS, help="Power label, e.g. 0p15.")
    parser.add_argument(
        "--rank",
        type=int,
        default=19,
        help="Fixed POD/ROM rank (default: 19, matching the notebook).",
    )
    parser.add_argument(
        "--model-form",
        type=str.upper,
        choices=("A", "AB", "AN", "ABN"),
        default="ABN",
        help=(
            "ROM drift terms: A, AB, AN, or ABN "
            "(default: ABN, matching main.ipynb)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
        help="Directory containing one subdirectory per power.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional experiment labels; default uses the notebook label list.",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=5_760,
        help="Samples per realization (default: 5760).",
    )
    parser.add_argument(
        "--regularization-count",
        type=int,
        default=10,
        help="Number of logarithmic drift-regularization candidates (default: 10).",
    )
    parser.add_argument(
        "--a-regularization",
        "--a-reg",
        type=float,
        default=0.0,
        help=(
            "Fixed A-operator regularization when N is grid-searched "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--b-regularization",
        "--b-reg",
        type=float,
        default=0.0,
        help="Fixed B-operator regularization (default: 0).",
    )
    parser.add_argument(
        "--n-regularization-min",
        "--n-reg-min",
        type=float,
        default=1.0,
        help="Smallest N-regularization grid value (default: 1).",
    )
    parser.add_argument(
        "--n-regularization-max",
        "--n-reg-max",
        type=float,
        default=1e10,
        help="Largest N-regularization grid value (default: 1e10).",
    )
    parser.add_argument(
        "--h-regularization",
        type=float,
        default=1e5,
        help="Diffusion regularization H_reg (default: 1e5).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="ROM noise multiplier (default: 1).",
    )
    parser.add_argument(
        "--input-amplitude",
        type=float,
        default=1.0,
        help="Cosine input amplitude (default: 1, matching the ABN paper-style run).",
    )
    parser.add_argument(
        "--input-frequency",
        type=float,
        default=7e6,
        help="Cosine input frequency in Hz (default: 7e6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ROM simulation seed (default: 42).",
    )
    parser.add_argument(
        "--psd-center-index",
        type=int,
        default=100,
        help="Spatial index used for the center-point PSD (default: 100).",
    )
    parser.add_argument(
        "--psd-target-segments",
        type=int,
        default=30,
        help=(
            "Target Welch segments for each trajectory PSD "
            "(default: 30, matching PSD.ipynb)."
        ),
    )
    parser.add_argument(
        "--psd-experimental-trajectories",
        type=int,
        default=10,
        help=(
            "Number of experimental trajectories used in the PSD comparison "
            "(default: 10; 0 omits experimental PSDs)."
        ),
    )
    parser.add_argument(
        "--psd-rom-trajectories",
        type=int,
        default=20,
        help=(
            "Number of fresh selected-ROM trajectories used in the PSD "
            "comparison (default: 20; 0 omits ROM PSDs)."
        ),
    )
    parser.add_argument(
        "--coordinate-modes",
        nargs="+",
        type=int,
        help=(
            "One-based POD mode numbers for the matched experimental/ROM "
            "trajectory plot (default: 1 2 3 5 10 20, restricted to rank)."
        ),
    )
    parser.add_argument(
        "--coordinate-rom-trajectories",
        type=int,
        default=5,
        help=(
            "Maximum ROM trajectories overlaid on all experimental "
            "realizations in the coordinate-ensemble plot (default: 5)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output folder (default: main_analysis_results/"
            "<power>_r<rank>_<model-form>)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a nonempty output directory.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_labels(power: str, labels: list[str], data_dir: Path) -> None:
    if not labels:
        raise ValueError("At least one experiment label is required.")
    if len(labels) != len(set(labels)):
        raise ValueError("Experiment labels must be unique.")
    missing = [
        data_dir / power / f"Q_1D_{power}vpp_{label}.h5"
        for label in labels
        if not (data_dir / power / f"Q_1D_{power}vpp_{label}.h5").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing source file(s):\n" + "\n".join(str(path) for path in missing)
        )


def compute_pod_basis(
    states: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the notebook's uncentered POD from every segmented realization."""
    n_space = states.shape[0]
    gram = np.zeros((n_space, n_space), dtype=np.float64)
    started = time.perf_counter()
    for realization in range(states.shape[1]):
        snapshot = np.asarray(states[:, realization, :], dtype=np.float64)
        gram += snapshot @ snapshot.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order[:rank]]

    # Stabilize the otherwise arbitrary signs for reproducible saved outputs.
    for mode in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0:
            basis[:, mode] *= -1

    singular_values = np.sqrt(eigenvalues)
    total_energy = float(eigenvalues.sum())
    if not np.isfinite(total_energy) or total_energy <= 0:
        raise ValueError("The snapshot matrix has no finite positive POD energy.")
    cumulative_energy = np.cumsum(eigenvalues) / total_energy
    print(
        f"POD basis computed in {time.perf_counter() - started:.1f} s; "
        f"rank-{rank} energy {cumulative_energy[rank - 1]:.6%}.",
        flush=True,
    )
    return basis, singular_values, cumulative_energy


def reduced_statistics(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ensemble mean and sample covariance at every relative time."""
    if states.shape[1] < 2:
        raise ValueError("At least two realizations are required for covariance.")
    mean = states.mean(axis=1)
    covariance = sto_opinf.page_cov(states, transpose_pages=True)
    return mean, covariance


def regularization_candidates(
    model_form: str,
    count: int,
    n_regularization_min: float = 1.0,
    n_regularization_max: float = 1e10,
) -> tuple[np.ndarray, str]:
    if "N" in model_form:
        return np.geomspace(
            n_regularization_min,
            n_regularization_max,
            count,
        ), "N"
    return np.logspace(-1, 5, count), "A"


def lambda_factory(
    model_form: str,
    candidates: np.ndarray,
    a_regularization: float = 0.0,
    b_regularization: float = 0.0,
) -> Callable[[int], np.ndarray]:
    is_bilinear = "N" in model_form
    has_input_term = "B" in model_form

    def get_lambda(index: int) -> np.ndarray:
        candidate = candidates[index]
        if is_bilinear and has_input_term:
            return np.array(
                [a_regularization, b_regularization, candidate]
            )
        if is_bilinear:
            return np.array([a_regularization, candidate])
        if has_input_term:
            return np.array([candidate, b_regularization])
        return np.array([candidate])

    return get_lambda


def make_rom_step(
    mass: np.ndarray,
    drift: np.ndarray,
    input_operator: np.ndarray,
    bilinear: np.ndarray,
    diffusion: np.ndarray,
    time_step: float,
    sigma: float,
) -> Callable[[np.ndarray, float, int, int, np.ndarray], np.ndarray]:
    """Create the implicit Euler-Maruyama step used throughout this workflow."""

    def step(
        state: np.ndarray,
        input_value: float,
        batch_size: int,
        step_index: int,
        noise: np.ndarray,
    ) -> np.ndarray:
        del batch_size, step_index
        lhs = mass - time_step * drift - time_step * bilinear * input_value
        stochastic_increment = np.sqrt(time_step) * diffusion * sigma @ noise
        deterministic_input = (
            time_step * input_operator * input_value
        ).reshape(-1, 1)
        return np.linalg.solve(
            lhs,
            state + deterministic_input + stochastic_increment,
        )

    return step


def run_compact_grid_search(
    reduced_mean: np.ndarray,
    reduced_covariance: np.ndarray,
    reduced_states: np.ndarray,
    input_signal: np.ndarray,
    time_step: float,
    model_form: str,
    candidates: np.ndarray,
    h_regularization: float,
    sigma: float,
    seed: int,
    a_regularization: float = 0.0,
    b_regularization: float = 0.0,
) -> dict[str, object]:
    """Run the notebook grid without retaining all candidate trajectories."""
    is_bilinear = "N" in model_form
    has_input_term = "B" in model_form
    get_lambda = lambda_factory(
        model_form,
        candidates,
        a_regularization,
        b_regularization,
    )
    n_candidates = len(candidates)
    n_realizations = reduced_states.shape[1]
    n_time = reduced_states.shape[2]
    initial_conditions = reduced_states[:, :, 0]

    candidate_means: list[np.ndarray | None] = [None] * n_candidates
    candidate_covariances: list[np.ndarray | None] = [None] * n_candidates
    candidate_operators: list[dict[str, np.ndarray] | None] = [None] * n_candidates
    mean_errors = np.full(n_candidates, np.nan)
    covariance_errors = np.full(n_candidates, np.nan)

    for index, candidate in enumerate(candidates):
        started = time.perf_counter()
        regularization = get_lambda(index)
        try:
            mass, drift, input_operator, bilinear = sto_opinf.infer_drift(
                reduced_mean,
                input_signal,
                time_step,
                is_bilinear,
                has_input_term,
                regularization,
            )
            diffusion, diffusion_covariance = sto_opinf.infer_diffusion(
                reduced_covariance,
                input_signal,
                time_step,
                drift,
                bilinear,
                h_regularization,
            )

            step = make_rom_step(
                mass,
                drift,
                input_operator,
                bilinear,
                diffusion,
                time_step,
                sigma,
            )
            np.random.seed(seed)
            model_mean, model_covariance, trajectory = sto_opinf.compute_model(
                step,
                initial_conditions,
                input_signal,
                n_realizations,
                n_realizations,
                diffusion.shape[1],
            )
            del trajectory

            mean_error = np.linalg.norm(reduced_mean - model_mean) / np.linalg.norm(
                reduced_mean
            )
            covariance_error = sto_opinf.page_norm(
                reduced_covariance - model_covariance
            ) / sto_opinf.page_norm(reduced_covariance)

            candidate_means[index] = model_mean
            candidate_covariances[index] = model_covariance
            candidate_operators[index] = {
                "Ehat": mass,
                "Ahat": drift,
                "Bhat": input_operator,
                "Nhat": bilinear,
                "Mhat": diffusion,
                "Khat": diffusion_covariance,
                "regularization": regularization,
            }
            mean_errors[index] = mean_error
            covariance_errors[index] = covariance_error
            print(
                f"Candidate {index + 1:>2}/{n_candidates}: "
                f"{candidate:.3e}, mean error={mean_error:.5e}, "
                f"covariance error={covariance_error:.5e} "
                f"({time.perf_counter() - started:.1f} s)",
                flush=True,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
            print(
                f"Candidate {index + 1:>2}/{n_candidates}: "
                f"{candidate:.3e} failed: {error}",
                flush=True,
            )

    finite = np.isfinite(mean_errors)
    if not finite.any():
        raise RuntimeError("Every regularization candidate failed.")
    selection_values = np.where(finite, mean_errors, np.inf)
    best_mean_index = int(np.argmin(selection_values))
    finite_covariance = np.isfinite(covariance_errors)
    best_covariance_index = int(
        np.argmin(np.where(finite_covariance, covariance_errors, np.inf))
    )

    return {
        "candidate_means": candidate_means,
        "candidate_covariances": candidate_covariances,
        "candidate_operators": candidate_operators,
        "mean_errors": mean_errors,
        "covariance_errors": covariance_errors,
        "best_mean_index": best_mean_index,
        "best_covariance_index": best_covariance_index,
    }


def time_resolved_errors(
    experimental_mean: np.ndarray,
    experimental_covariance: np.ndarray,
    model_mean: np.ndarray,
    model_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = np.finfo(np.float64).eps
    mean_denominator = np.maximum(
        np.linalg.norm(experimental_mean, axis=0), epsilon
    )
    covariance_denominator = np.maximum(
        np.linalg.norm(experimental_covariance, axis=(0, 1)), epsilon
    )
    mean_error = (
        np.linalg.norm(experimental_mean - model_mean, axis=0) / mean_denominator
    )
    covariance_error = (
        np.linalg.norm(
            experimental_covariance - model_covariance, axis=(0, 1)
        )
        / covariance_denominator
    )
    return mean_error, covariance_error


def compute_center_trajectory_psds(
    experimental_signals: np.ndarray,
    model_signals: np.ndarray,
    sample_rate: float,
    target_segments: int,
) -> dict[str, np.ndarray | float | int]:
    """Compute one Welch PSD per experimental and ROM center trajectory."""
    experimental_signals = np.asarray(experimental_signals)
    model_signals = np.asarray(model_signals)
    if experimental_signals.ndim != 2 or model_signals.ndim != 2:
        raise ValueError("Experimental and ROM signals must be trajectory-by-time arrays.")
    if experimental_signals.shape[1] != model_signals.shape[1]:
        raise ValueError("Experimental and ROM trajectories must have equal lengths.")
    if experimental_signals.shape[0] + model_signals.shape[0] == 0:
        raise ValueError("At least one experimental or ROM trajectory is required.")
    if target_segments < 1:
        raise ValueError("target_segments must be positive.")

    signal_length = experimental_signals.shape[1]
    nperseg = min(
        signal_length,
        max(128, int(2 * signal_length / (target_segments + 1))),
    )
    if nperseg % 2:
        nperseg -= 1

    frequency = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)

    def compute(signals: np.ndarray) -> np.ndarray:
        if signals.shape[0] == 0:
            return np.empty((0, frequency.size), dtype=np.float64)
        group_frequency, densities = PSD_utils.welch_psd_f(
            signals,
            fs=sample_rate,
            nperseg=nperseg,
        )
        if not np.array_equal(frequency, group_frequency):
            raise RuntimeError("Experimental and ROM PSD frequency grids differ.")
        return densities

    experimental_psds = compute(experimental_signals)
    model_psds = compute(model_signals)
    return {
        "frequency": frequency,
        "experimental_psds": experimental_psds,
        "model_psds": model_psds,
        "sample_rate": float(sample_rate),
        "nperseg": nperseg,
        "target_segments": target_segments,
    }


def center_mode_components(
    center_basis_row: np.ndarray,
    reduced_trajectories: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split rank-r center signals into POD mode 1 and all other modes."""
    center_basis_row = np.asarray(center_basis_row, dtype=np.float64)
    reduced_trajectories = np.asarray(reduced_trajectories, dtype=np.float64)
    if center_basis_row.ndim != 1:
        raise ValueError("center_basis_row must be one-dimensional.")
    if reduced_trajectories.ndim != 3:
        raise ValueError(
            "Reduced trajectories must have shape [mode, trajectory, time]."
        )
    if reduced_trajectories.shape[0] != center_basis_row.size:
        raise ValueError("Basis row and reduced trajectory rank do not match.")
    if center_basis_row.size < 1:
        raise ValueError("At least one POD mode is required.")

    mode_1 = center_basis_row[0] * reduced_trajectories[0]
    without_mode_1 = np.einsum(
        "r,rlt->lt",
        center_basis_row[1:],
        reduced_trajectories[1:],
    )
    return {
        "center": mode_1 + without_mode_1,
        "center_without_mode_1": without_mode_1,
        "mode_1_at_center": mode_1,
    }


def compute_center_mode_component_psds(
    center_basis_row: np.ndarray,
    experimental_reduced_trajectories: np.ndarray,
    model_reduced_trajectories: np.ndarray,
    sample_rate: float,
    target_segments: int,
) -> dict[str, dict[str, np.ndarray | float | int]]:
    """Compute PSDs for the rank-r center signal and its mode-1 split."""
    experimental = center_mode_components(
        center_basis_row, experimental_reduced_trajectories
    )
    model = center_mode_components(center_basis_row, model_reduced_trajectories)
    return {
        name: compute_center_trajectory_psds(
            experimental[name],
            model[name],
            sample_rate,
            target_segments,
        )
        for name in experimental
    }


def summarize_center_mode_component_psds(
    component_results: dict[str, dict[str, np.ndarray | float | int]],
) -> dict[str, dict[str, float | None]]:
    """Integrate mean component PSDs for compact JSON diagnostics."""
    summary: dict[str, dict[str, float | None]] = {}
    for name, results in component_results.items():
        frequency = np.asarray(results["frequency"], dtype=np.float64)

        def mean_power(key: str) -> float | None:
            densities = np.asarray(results[key], dtype=np.float64)
            if densities.shape[0] == 0:
                return None
            return float(np.trapezoid(densities.mean(axis=0), frequency))

        experimental_power = mean_power("experimental_psds")
        model_power = mean_power("model_psds")
        ratio = (
            None
            if experimental_power is None
            or model_power is None
            or experimental_power <= 0
            else model_power / experimental_power
        )
        summary[name] = {
            "experimental_mean_power": experimental_power,
            "rom_mean_power": model_power,
            "rom_over_experimental": ratio,
        }
    return summary


def save_center_psd_comparison_plot(
    psd_results: dict[str, np.ndarray | float | int],
    center_index: int,
    center_x: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot mean trajectory PSDs with central 10--90% trajectory bands."""
    frequency = np.asarray(psd_results["frequency"])
    positive_frequency = frequency > 0
    tiny = np.finfo(np.float64).tiny

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = []
    for key, label, color in (
        ("experimental_psds", "Experimental data", "C0"),
        ("model_psds", "ROM data", "C1"),
    ):
        densities = np.asarray(psd_results[key])
        counts.append(f"{densities.shape[0]} {label.lower()}")
        if densities.shape[0] == 0:
            continue
        mean = densities.mean(axis=0)
        lower, upper = np.quantile(densities, (0.1, 0.9), axis=0)
        ax.plot(
            frequency[positive_frequency],
            np.log10(np.maximum(mean[positive_frequency], tiny)),
            color=color,
            linewidth=1.2,
            label=label,
        )
        ax.fill_between(
            frequency[positive_frequency],
            np.log10(np.maximum(lower[positive_frequency], tiny)),
            np.log10(np.maximum(upper[positive_frequency], tiny)),
            color=color,
            alpha=0.2,
        )
    ax.set_xscale("log")
    ax.set_xlim(frequency[positive_frequency][0], frequency[-1])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\log_{10}(\mathrm{PSD})$ [$\mu$m$^2$/Hz]")
    ax.set_title(
        f"Center-point trajectory PSDs ({', '.join(counts)}; 10--90% bands)\n"
        rf"index {center_index}, x={center_x:.3g} $\mu$m"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_center_mode_component_psd_plot(
    component_results: dict[str, dict[str, np.ndarray | float | int]],
    center_index: int,
    center_x: float,
    rank: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the center PSD before and after separating POD mode 1."""
    panels = (
        ("center", f"center (rank-{rank} reconstruction)"),
        ("center_without_mode_1", "center without POD mode 1"),
        ("mode_1_at_center", "POD mode 1 contribution at center"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    tiny = np.finfo(np.float64).tiny
    for axis, (name, title) in zip(axes, panels):
        results = component_results[name]
        frequency = np.asarray(results["frequency"], dtype=np.float64)
        positive = frequency > 0
        for key, label, color in (
            ("experimental_psds", "Experimental data", "C0"),
            ("model_psds", "ROM data", "C1"),
        ):
            densities = np.asarray(results[key], dtype=np.float64)
            if densities.shape[0] == 0:
                continue
            mean = densities.mean(axis=0)
            lower, upper = np.quantile(densities, (0.1, 0.9), axis=0)
            axis.loglog(
                frequency[positive],
                np.maximum(mean[positive], tiny),
                color=color,
                linewidth=1.2,
                label=label,
            )
            axis.fill_between(
                frequency[positive],
                np.maximum(lower[positive], tiny),
                np.maximum(upper[positive], tiny),
                color=color,
                alpha=0.2,
            )
        axis.set_ylabel(r"PSD [$\mu$m$^2$/Hz]")
        axis.set_title(title)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(
        "Center-point PSD decomposition\n"
        rf"index {center_index}, x={center_x:.3g} $\mu$m"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_coordinate_trajectory_plot(
    relative_time: np.ndarray,
    experimental_trajectory: np.ndarray,
    model_trajectory: np.ndarray | None,
    modes: list[int],
    experimental_index: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot matched experimental and ROM reduced-coordinate trajectories."""
    experimental_trajectory = np.asarray(experimental_trajectory, dtype=np.float64)
    if experimental_trajectory.ndim != 2:
        raise ValueError("Experimental coordinates must have shape [mode, time].")
    if model_trajectory is not None:
        model_trajectory = np.asarray(model_trajectory, dtype=np.float64)
        if model_trajectory.shape != experimental_trajectory.shape:
            raise ValueError("Experimental and ROM coordinate shapes must match.")

    indices = np.asarray(modes, dtype=int) - 1
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(10, 2.1 * len(modes)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for axis, mode, index in zip(axes[:, 0], modes, indices):
        axis.plot(
            relative_time,
            experimental_trajectory[index],
            color="0.2",
            linewidth=0.8,
            label="Experimental" if mode == modes[0] else None,
        )
        if model_trajectory is not None:
            axis.plot(
                relative_time,
                model_trajectory[index],
                color="C3",
                linewidth=0.8,
                label="ROM" if mode == modes[0] else None,
            )
        axis.set_ylabel(f"POD {mode}\n[$\\mu$m]")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time since segment start [s]")
    fig.suptitle(f"Reduced-coordinate trajectories · realization {experimental_index}")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_coordinate_ensemble_plot(
    relative_time: np.ndarray,
    experimental_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    modes: list[int],
    output_path: Path,
    dpi: int,
) -> None:
    """Overlay every experimental realization and a small ROM ensemble."""
    experimental_trajectories = np.asarray(
        experimental_trajectories, dtype=np.float64
    )
    model_trajectories = np.asarray(model_trajectories, dtype=np.float64)
    if experimental_trajectories.ndim != 3 or model_trajectories.ndim != 3:
        raise ValueError(
            "Coordinate ensembles must have shape [mode, trajectory, time]."
        )
    if (
        experimental_trajectories.shape[0] != model_trajectories.shape[0]
        or experimental_trajectories.shape[2] != model_trajectories.shape[2]
    ):
        raise ValueError("Experimental and ROM coordinate ensemble shapes disagree.")
    if experimental_trajectories.shape[1] < 1:
        raise ValueError("At least one experimental realization is required.")

    indices = np.asarray(modes, dtype=int) - 1
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(10, 2.1 * len(modes)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    experimental_alpha = min(
        0.25,
        max(0.03, 10.0 / experimental_trajectories.shape[1]),
    )
    for axis, mode, index in zip(axes[:, 0], modes, indices):
        axis.plot(
            relative_time,
            experimental_trajectories[index].T,
            color="C0",
            linewidth=0.35,
            alpha=experimental_alpha,
        )
        if model_trajectories.shape[1]:
            axis.plot(
                relative_time,
                model_trajectories[index].T,
                color="C3",
                linewidth=0.8,
                alpha=0.75,
            )
        axis.set_ylabel(f"POD {mode}\n[$\\mu$m]")
        axis.grid(alpha=0.2)

    # Proxy lines keep the legend compact instead of adding one entry per path.
    axes[0, 0].plot(
        [], [], color="C0", linewidth=1.2,
        label=f"Experimental ({experimental_trajectories.shape[1]})",
    )
    if model_trajectories.shape[1]:
        axes[0, 0].plot(
            [], [], color="C3", linewidth=1.2,
            label=f"ROM ({model_trajectories.shape[1]})",
        )
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time since segment start [s]")
    fig.suptitle("Reduced-coordinate realization ensemble")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_experimental_mean_plot(
    relative_time: np.ndarray,
    x: np.ndarray,
    experimental_mean: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.pcolormesh(
        relative_time,
        x,
        experimental_mean,
        shading="auto",
        cmap="jet",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$x$ [$\mu$m]")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"Surface displacement [$\mu$m]")
    ax.set_title("Experimental ensemble mean")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_regularization_plot(
    candidates: np.ndarray,
    mean_errors: np.ndarray,
    covariance_errors: np.ndarray,
    best_mean_index: int,
    operator_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, values, color, label in zip(
        axes,
        (mean_errors, covariance_errors),
        ("royalblue", "tomato"),
        ("Mean error", "Covariance error"),
    ):
        valid = np.isfinite(values) & (values > 0)
        ax.loglog(
            candidates[valid],
            values[valid],
            "-o",
            color=color,
            markerfacecolor="none",
        )
        if np.isfinite(values[best_mean_index]):
            ax.scatter(
                candidates[best_mean_index],
                values[best_mean_index],
                color="black",
                marker="*",
                s=90,
                zorder=3,
                label="selected by mean error",
            )
        ax.set_xlabel(rf"${operator_name}$ regularization")
        ax.set_ylabel(label)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
    fig.suptitle("Regularization-grid errors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_relative_error_plot(
    relative_time: np.ndarray,
    mean_error: np.ndarray,
    covariance_error: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    tiny = np.finfo(np.float64).tiny
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(
        relative_time,
        np.maximum(mean_error, tiny),
        "--",
        color="C0",
        linewidth=1.5,
        label="mean error",
    )
    ax.semilogy(
        relative_time,
        np.maximum(covariance_error, tiny),
        "-",
        color="black",
        linewidth=1.5,
        label="covariance error",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative error")
    ax.set_xlim(relative_time[0], relative_time[-1])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Reduced experimental and ROM moment errors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reduced_mean_plot(
    relative_time: np.ndarray,
    experimental_mean: np.ndarray,
    model_mean: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    difference = np.abs(experimental_mean - model_mean)
    extent = (
        float(relative_time[0]),
        float(relative_time[-1]),
        1,
        experimental_mean.shape[0],
    )
    limit = max(
        float(np.max(np.abs(experimental_mean))),
        float(np.max(np.abs(model_mean))),
        np.finfo(float).eps,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    shared_image = None
    for ax, values, title in zip(
        axes[:2],
        (experimental_mean, model_mean),
        ("Experiment", "ROM"),
    ):
        shared_image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title)
    assert shared_image is not None
    fig.colorbar(shared_image, ax=axes[:2], label="Reduced mean coefficient")
    error_image = axes[2].imshow(
        difference,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[2].set_title("Absolute difference")
    fig.colorbar(error_image, ax=axes[2], label="Absolute error")
    for ax in axes:
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("POD mode")
    fig.suptitle("Reduced-space ensemble mean")
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.14, top=0.82, wspace=0.22)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reduced_covariance_plot(
    relative_time: np.ndarray,
    experimental_covariance: np.ndarray,
    model_covariance: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    indices = np.unique(np.linspace(0, len(relative_time) - 1, 3, dtype=int))
    fig, axes = plt.subplots(
        3,
        len(indices),
        figsize=(4.2 * len(indices), 10),
        squeeze=False,
    )
    for column, time_index in enumerate(indices):
        limit = max(
            float(np.max(np.abs(experimental_covariance[:, :, time_index]))),
            float(np.max(np.abs(model_covariance[:, :, time_index]))),
            np.finfo(float).eps,
        )
        for row, covariance in enumerate(
            (experimental_covariance, model_covariance)
        ):
            image = axes[row, column].imshow(
                covariance[:, :, time_index],
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
        difference = np.abs(
            experimental_covariance[:, :, time_index]
            - model_covariance[:, :, time_index]
        )
        difference_image = axes[2, column].imshow(
            difference,
            origin="lower",
            cmap="magma",
        )
        fig.colorbar(
            difference_image,
            ax=axes[2, column],
            fraction=0.046,
            pad=0.04,
        )
        axes[0, column].set_title(f"t = {relative_time[time_index]:.4g} s")

    for row, label in enumerate(("Experiment", "ROM", "Absolute difference")):
        axes[row, 0].set_ylabel(f"{label}\nPOD mode")
    for ax in axes.flat:
        ax.set_xlabel("POD mode")
    fig.suptitle("Reduced-space covariance")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_full_mean_comparison_plot(
    relative_time: np.ndarray,
    x: np.ndarray,
    experimental_mean: np.ndarray,
    model_mean: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    epsilon = np.finfo(np.float64).eps
    relative_error = np.abs(experimental_mean - model_mean) / np.maximum(
        np.abs(experimental_mean), epsilon
    )
    # A few points where the experimental mean is almost zero can make the
    # color range enormous. Keep the exact relative errors, but saturate the
    # upper 1% in this diagnostic so the spatial/temporal structure is visible.
    finite_relative_error = relative_error[np.isfinite(relative_error)]
    error_color_limit = float(np.percentile(finite_relative_error, 99.0))
    error_color_limit = max(error_color_limit, epsilon)
    limit_min = min(float(experimental_mean.min()), float(model_mean.min()))
    limit_max = max(float(experimental_mean.max()), float(model_mean.max()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    shared_image = None
    for ax, values, title in zip(
        axes[:2],
        (experimental_mean, model_mean),
        ("Experiment", "ROM"),
    ):
        shared_image = ax.pcolormesh(
            relative_time,
            x,
            values,
            shading="auto",
            cmap="jet",
            vmin=limit_min,
            vmax=limit_max,
        )
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
    assert shared_image is not None
    fig.colorbar(
        shared_image,
        ax=axes[:2],
        label=r"Surface displacement [$\mu$m]",
    )
    error_image = axes[2].pcolormesh(
        relative_time,
        x,
        relative_error,
        shading="auto",
        cmap="hot",
        vmin=0.0,
        vmax=error_color_limit,
    )
    axes[2].set_title("Pointwise relative error (colors clipped at P99)")
    axes[2].set_xlabel("Time [s]")
    axes[0].set_ylabel(r"$x$ [$\mu$m]")
    fig.colorbar(error_image, ax=axes[2], label="Relative error")
    fig.suptitle("Full-space ensemble mean")
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.15, top=0.82, wspace=0.25)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_final_experimental_statistics(
    relative_time: np.ndarray,
    reduced_mean: np.ndarray,
    reduced_covariance: np.ndarray,
) -> str:
    mean_text = np.array2string(
        reduced_mean[:, -1],
        precision=8,
        suppress_small=False,
        threshold=np.inf,
        max_line_width=220,
    )
    covariance_text = np.array2string(
        reduced_covariance[:, :, -1],
        precision=8,
        suppress_small=False,
        threshold=np.inf,
        max_line_width=220,
    )
    return (
        f"Final experimental reduced statistics at t={relative_time[-1]:.9g} s\n"
        f"\nMean vector Er_data[:, -1] (shape {reduced_mean[:, -1].shape}):\n"
        f"{mean_text}\n"
        f"\nCovariance matrix Cr_data[:, :, -1] "
        f"(shape {reduced_covariance[:, :, -1].shape}):\n"
        f"{covariance_text}\n"
    )


def main() -> None:
    args = parse_args()
    if args.rank < 2:
        raise ValueError("--rank must be at least 2 for covariance modeling.")
    if args.rank > 200:
        raise ValueError("--rank cannot exceed the 200-point spatial dimension.")
    if args.split_size < 7:
        raise ValueError("--split-size must be at least 7 for the derivative filters.")
    if args.regularization_count < 1:
        raise ValueError("--regularization-count must be positive.")
    if (
        args.a_regularization < 0
        or args.b_regularization < 0
        or args.h_regularization < 0
        or args.sigma < 0
    ):
        raise ValueError(
            "A, B, and H regularization and --sigma must be nonnegative."
        )
    if args.n_regularization_min <= 0 or args.n_regularization_max <= 0:
        raise ValueError("N-regularization grid endpoints must be positive.")
    if args.n_regularization_min > args.n_regularization_max:
        raise ValueError(
            "--n-regularization-min cannot exceed --n-regularization-max."
        )
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    if args.psd_target_segments < 1:
        raise ValueError("--psd-target-segments must be positive.")
    if args.psd_experimental_trajectories < 0 or args.psd_rom_trajectories < 0:
        raise ValueError("PSD trajectory counts must be nonnegative.")
    if args.coordinate_rom_trajectories < 0:
        raise ValueError("--coordinate-rom-trajectories must be nonnegative.")
    if args.psd_experimental_trajectories + args.psd_rom_trajectories == 0:
        raise ValueError("At least one PSD trajectory count must be positive.")
    coordinate_modes = args.coordinate_modes or [
        mode for mode in (1, 2, 3, 5, 10, 20) if mode <= args.rank
    ]
    if (
        not coordinate_modes
        or len(coordinate_modes) != len(set(coordinate_modes))
        or any(mode < 1 or mode > args.rank for mode in coordinate_modes)
    ):
        raise ValueError(
            "--coordinate-modes must be unique one-based mode numbers "
            f"between 1 and rank {args.rank}."
        )

    configure_plotting()
    data_dir = args.data_dir.expanduser()
    labels = args.labels or POWER_LABELS[args.power]
    validate_labels(args.power, labels, data_dir)
    output_dir = args.output_dir or Path("main_analysis_results") / (
        f"{args.power}_r{args.rank}_{args.model_form}"
    )
    output_dir = output_dir.expanduser()
    prepare_output_directory(output_dir, args.overwrite)

    print(
        f"Loading {args.power} experiments {labels} from {data_dir.resolve()}...",
        flush=True,
    )
    dataset = sto_opinf.QDataset.load(
        args.power,
        labels,
        str(data_dir),
    ).preprocess(split_size=args.split_size)
    states = dataset.Qstate_all
    relative_time = dataset.tt
    if states is None or relative_time is None:
        raise RuntimeError("Dataset preprocessing did not produce the expected arrays.")
    x = np.asarray(dataset.x, dtype=np.float64)
    raw_time = np.asarray(dataset.t, dtype=np.float64)
    n_space, n_realizations, n_time = states.shape
    if args.rank > n_space:
        raise ValueError(f"--rank cannot exceed spatial dimension {n_space}.")
    if n_realizations < 2:
        raise ValueError("At least two segmented realizations are required.")
    if args.psd_experimental_trajectories > n_realizations:
        raise ValueError(
            "--psd-experimental-trajectories cannot exceed the available "
            f"{n_realizations} segmented realizations."
        )
    if not 0 <= args.psd_center_index < n_space:
        raise ValueError(
            f"--psd-center-index must be between 0 and {n_space - 1}."
        )
    print(
        f"Preprocessed state shape: {states.shape} "
        "(space, realization, relative time).",
        flush=True,
    )

    experimental_rng = np.random.default_rng(args.seed)
    experimental_psd_indices = np.sort(
        experimental_rng.choice(
            n_realizations,
            size=args.psd_experimental_trajectories,
            replace=False,
        )
    )
    experimental_center_trajectories = np.asarray(
        states[args.psd_center_index, experimental_psd_indices, :],
        dtype=np.float64,
    ).copy()

    # The concatenated state is independent; release raw arrays and split views.
    dataset.Q.clear()
    if dataset.Q_split is not None:
        dataset.Q_split.clear()
    dataset.Qstate_all = None
    del dataset
    gc.collect()

    experimental_full_mean = states.mean(axis=1)
    basis, singular_values, cumulative_energy = compute_pod_basis(
        states,
        args.rank,
    )
    reduced_states = (
        basis.T @ states.reshape(n_space, n_realizations * n_time)
    ).reshape(args.rank, n_realizations, n_time)
    del states
    gc.collect()
    experimental_reduced_mean, experimental_reduced_covariance = (
        reduced_statistics(reduced_states)
    )

    input_signal = args.input_amplitude * np.cos(
        2 * np.pi * args.input_frequency * relative_time
    )
    time_step = float(raw_time[1] - raw_time[0])
    candidates, regularized_operator = regularization_candidates(
        args.model_form,
        args.regularization_count,
        args.n_regularization_min,
        args.n_regularization_max,
    )
    print(
        f"Running {args.model_form} model grid with {len(candidates)} "
        f"{regularized_operator}-regularization candidates; "
        f"A_reg={args.a_regularization:.3e}, "
        f"B_reg={args.b_regularization:.3e}, "
        f"H_reg={args.h_regularization:.3e}...",
        flush=True,
    )
    grid_results = run_compact_grid_search(
        experimental_reduced_mean,
        experimental_reduced_covariance,
        reduced_states,
        input_signal,
        time_step,
        args.model_form,
        candidates,
        args.h_regularization,
        args.sigma,
        args.seed,
        args.a_regularization,
        args.b_regularization,
    )
    best_mean_index = int(grid_results["best_mean_index"])
    best_covariance_index = int(grid_results["best_covariance_index"])
    candidate_means = grid_results["candidate_means"]
    candidate_covariances = grid_results["candidate_covariances"]
    candidate_operators = grid_results["candidate_operators"]
    model_reduced_mean = candidate_means[best_mean_index]
    model_reduced_covariance = candidate_covariances[best_mean_index]
    operators = candidate_operators[best_mean_index]
    assert isinstance(model_reduced_mean, np.ndarray)
    assert isinstance(model_reduced_covariance, np.ndarray)
    assert isinstance(operators, dict)

    mean_errors = np.asarray(grid_results["mean_errors"], dtype=np.float64)
    covariance_errors = np.asarray(
        grid_results["covariance_errors"], dtype=np.float64
    )
    mean_error_time, covariance_error_time = time_resolved_errors(
        experimental_reduced_mean,
        experimental_reduced_covariance,
        model_reduced_mean,
        model_reduced_covariance,
    )
    model_full_mean = basis @ model_reduced_mean

    first_mode_constant_overlap = float(
        abs(basis[:, 0].sum()) / np.sqrt(n_space)
    )
    first_mode_energy_fraction = float(cumulative_energy[0])
    print(
        "POD mode 1 diagnostics: "
        f"|constant overlap|={first_mode_constant_overlap:.9f}, "
        f"snapshot-energy fraction={first_mode_energy_fraction:.9%}.",
        flush=True,
    )

    rom_initial_rng = np.random.default_rng(args.seed + 1)
    rom_initial_indices = rom_initial_rng.choice(
        n_realizations,
        size=args.psd_rom_trajectories,
        replace=args.psd_rom_trajectories > n_realizations,
    )
    if args.psd_rom_trajectories:
        selected_step = make_rom_step(
            operators["Ehat"],
            operators["Ahat"],
            operators["Bhat"],
            operators["Nhat"],
            operators["Mhat"],
            time_step,
            args.sigma,
        )
        np.random.seed(args.seed + 1)
        model_reduced_trajectories = sto_opinf.stepSDE(
            selected_step,
            reduced_states[:, rom_initial_indices, 0],
            input_signal,
            args.psd_rom_trajectories,
            operators["Mhat"].shape[1],
        )
        model_center_trajectories = np.einsum(
            "r,rlt->lt",
            basis[args.psd_center_index, :],
            model_reduced_trajectories,
        )
    else:
        model_reduced_trajectories = np.empty(
            (args.rank, 0, n_time), dtype=np.float64
        )
        model_center_trajectories = np.empty((0, n_time), dtype=np.float64)

    psd_results = compute_center_trajectory_psds(
        experimental_center_trajectories,
        model_center_trajectories,
        sample_rate=1.0 / time_step,
        target_segments=args.psd_target_segments,
    )
    center_mode_psd_results = compute_center_mode_component_psds(
        basis[args.psd_center_index, :],
        reduced_states[:, experimental_psd_indices, :],
        model_reduced_trajectories,
        sample_rate=1.0 / time_step,
        target_segments=args.psd_target_segments,
    )
    center_mode_psd_summary = summarize_center_mode_component_psds(
        center_mode_psd_results
    )
    coordinate_rom_trajectory_count = min(
        args.coordinate_rom_trajectories,
        model_reduced_trajectories.shape[1],
    )
    psd_frequency = np.asarray(psd_results["frequency"])
    print(
        "Center PSD comparison uses "
        f"{args.psd_experimental_trajectories} experimental and "
        f"{args.psd_rom_trajectories} fresh ROM trajectories, "
        f"target_segments={args.psd_target_segments}, "
        f"nperseg={psd_results['nperseg']}, frequency range "
        f"{psd_frequency[1]:.6g}-{psd_frequency[-1]:.6g} Hz.",
        flush=True,
    )

    print(
        f"Best mean error: {mean_errors[best_mean_index]:.6e} at "
        f"{regularized_operator}_reg={candidates[best_mean_index]:.6e}\n"
        f"Covariance error for selected model: "
        f"{covariance_errors[best_mean_index]:.6e}\n"
        f"Best covariance error anywhere in grid: "
        f"{covariance_errors[best_covariance_index]:.6e} at "
        f"{regularized_operator}_reg={candidates[best_covariance_index]:.6e}\n"
        f"Maximum time-resolved mean error: {mean_error_time.max():.6e}\n"
        f"Maximum time-resolved covariance error: "
        f"{covariance_error_time.max():.6e}",
        flush=True,
    )

    final_statistics_text = format_final_experimental_statistics(
        relative_time,
        experimental_reduced_mean,
        experimental_reduced_covariance,
    )
    print("\n" + final_statistics_text, flush=True)
    (output_dir / "final_experimental_reduced_statistics.txt").write_text(
        final_statistics_text
    )

    metrics = {
        "power": args.power,
        "labels": labels,
        "rank": args.rank,
        "model_form": args.model_form,
        "split_size": args.split_size,
        "n_realizations": n_realizations,
        "n_time": n_time,
        "time_step": time_step,
        "sigma": args.sigma,
        "input_amplitude": args.input_amplitude,
        "input_frequency": args.input_frequency,
        "seed": args.seed,
        "a_regularization": args.a_regularization,
        "b_regularization": args.b_regularization,
        "n_regularization_min": args.n_regularization_min,
        "n_regularization_max": args.n_regularization_max,
        "h_regularization": args.h_regularization,
        "regularized_operator": regularized_operator,
        "regularization_candidates": candidates.tolist(),
        "selected_regularization": float(candidates[best_mean_index]),
        "selected_regularization_vector": np.asarray(
            operators["regularization"]
        ).tolist(),
        "best_mean_error": float(mean_errors[best_mean_index]),
        "selected_model_covariance_error": float(
            covariance_errors[best_mean_index]
        ),
        "best_covariance_regularization": float(
            candidates[best_covariance_index]
        ),
        "best_covariance_error": float(covariance_errors[best_covariance_index]),
        "maximum_time_resolved_mean_error": float(mean_error_time.max()),
        "maximum_time_resolved_covariance_error": float(
            covariance_error_time.max()
        ),
        "rank_energy_fraction": float(cumulative_energy[args.rank - 1]),
        "first_mode_constant_overlap_absolute": first_mode_constant_overlap,
        "first_mode_snapshot_energy_fraction": first_mode_energy_fraction,
        "first_mode_center_basis_weight": float(
            basis[args.psd_center_index, 0]
        ),
        "psd_center_index": args.psd_center_index,
        "psd_center_x_micrometers": float(x[args.psd_center_index]),
        "psd_experimental_trajectory_count": args.psd_experimental_trajectories,
        "psd_rom_trajectory_count": args.psd_rom_trajectories,
        "psd_aggregation": "mean_of_per_trajectory_psds",
        "psd_variation_band": "10th_to_90th_percentile",
        "psd_density": "frequency",
        "psd_target_segments": int(psd_results["target_segments"]),
        "psd_nperseg": int(psd_results["nperseg"]),
        "psd_sample_rate_hz": float(psd_results["sample_rate"]),
        "psd_lowest_nonzero_frequency_hz": float(psd_frequency[1]),
        "psd_nyquist_frequency_hz": float(psd_frequency[-1]),
        "center_mode_psd_decomposition": center_mode_psd_summary,
        "coordinate_trajectory_modes": coordinate_modes,
        "coordinate_ensemble_experimental_trajectory_count": n_realizations,
        "coordinate_ensemble_rom_trajectory_count": (
            coordinate_rom_trajectory_count
        ),
        "coordinate_trajectory_realization_index": (
            int(rom_initial_indices[0])
            if len(rom_initial_indices)
            else int(experimental_psd_indices[0])
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    np.savez_compressed(
        output_dir / "center_psd_comparison.npz",
        frequency_hz=psd_frequency,
        experimental_trajectory_psds=psd_results["experimental_psds"],
        rom_trajectory_psds=psd_results["model_psds"],
        experimental_trajectory_indices=experimental_psd_indices,
        rom_initial_condition_indices=rom_initial_indices,
        center_index=args.psd_center_index,
        center_x_micrometers=x[args.psd_center_index],
        sample_rate_hz=psd_results["sample_rate"],
        nperseg=psd_results["nperseg"],
        target_segments=psd_results["target_segments"],
        aggregation="mean_of_per_trajectory_psds",
        variation_band="10th_to_90th_percentile",
        density="frequency",
    )

    component_archive = {
        "frequency_hz": np.asarray(
            center_mode_psd_results["center"]["frequency"]
        ),
        "center_basis_row": basis[args.psd_center_index, :],
        "experimental_trajectory_indices": experimental_psd_indices,
        "rom_initial_condition_indices": rom_initial_indices,
        "center_index": args.psd_center_index,
        "center_x_micrometers": x[args.psd_center_index],
        "first_mode_constant_overlap_absolute": first_mode_constant_overlap,
        "first_mode_snapshot_energy_fraction": first_mode_energy_fraction,
        "sample_rate_hz": center_mode_psd_results["center"]["sample_rate"],
        "nperseg": center_mode_psd_results["center"]["nperseg"],
        "target_segments": center_mode_psd_results["center"]["target_segments"],
        "aggregation": "mean_of_per_trajectory_psds",
        "variation_band": "10th_to_90th_percentile",
        "density": "frequency",
    }
    for component, results in center_mode_psd_results.items():
        component_archive[f"experimental_{component}_psds"] = results[
            "experimental_psds"
        ]
        component_archive[f"rom_{component}_psds"] = results["model_psds"]
    np.savez_compressed(
        output_dir / "center_mode1_psd_comparison.npz",
        **component_archive,
    )

    coordinate_reference_index = (
        int(rom_initial_indices[0])
        if len(rom_initial_indices)
        else int(experimental_psd_indices[0])
    )
    coordinate_model_trajectory = (
        model_reduced_trajectories[:, 0, :]
        if model_reduced_trajectories.shape[1]
        else None
    )
    np.savez_compressed(
        output_dir / "coordinate_trajectories.npz",
        relative_time=relative_time,
        modes=np.asarray(coordinate_modes, dtype=np.int64),
        experimental_trajectory=reduced_states[
            :, coordinate_reference_index, :
        ],
        rom_trajectory=(
            coordinate_model_trajectory
            if coordinate_model_trajectory is not None
            else np.empty((args.rank, 0), dtype=np.float64)
        ),
        experimental_realization_index=coordinate_reference_index,
        rom_initial_condition_index=(
            coordinate_reference_index
            if coordinate_model_trajectory is not None
            else -1
        ),
        coordinate_units="micrometers",
    )

    np.savez_compressed(
        output_dir / "analysis_results.npz",
        x=x,
        relative_time=relative_time,
        basis=basis,
        singular_values=singular_values,
        cumulative_energy_fraction=cumulative_energy,
        experimental_full_mean=experimental_full_mean,
        model_full_mean=model_full_mean,
        experimental_reduced_mean=experimental_reduced_mean,
        model_reduced_mean=model_reduced_mean,
        experimental_reduced_covariance=experimental_reduced_covariance,
        model_reduced_covariance=model_reduced_covariance,
        mean_error_time=mean_error_time,
        covariance_error_time=covariance_error_time,
        regularization_candidates=candidates,
        regularization_mean_errors=mean_errors,
        regularization_covariance_errors=covariance_errors,
        selected_regularization_vector=operators["regularization"],
        a_regularization=args.a_regularization,
        b_regularization=args.b_regularization,
        n_regularization_min=args.n_regularization_min,
        n_regularization_max=args.n_regularization_max,
        h_regularization=args.h_regularization,
        Ehat=operators["Ehat"],
        Ahat=operators["Ahat"],
        Bhat=operators["Bhat"],
        Nhat=operators["Nhat"],
        Mhat=operators["Mhat"],
        Khat=operators["Khat"],
    )

    save_center_psd_comparison_plot(
        psd_results,
        args.psd_center_index,
        float(x[args.psd_center_index]),
        output_dir / "center_psd_comparison.png",
        args.dpi,
    )
    save_center_mode_component_psd_plot(
        center_mode_psd_results,
        args.psd_center_index,
        float(x[args.psd_center_index]),
        args.rank,
        output_dir / "center_mode1_psd_comparison.png",
        args.dpi,
    )
    save_coordinate_trajectory_plot(
        relative_time,
        reduced_states[:, coordinate_reference_index, :],
        coordinate_model_trajectory,
        coordinate_modes,
        coordinate_reference_index,
        output_dir / "coordinate_trajectories.png",
        args.dpi,
    )
    save_coordinate_ensemble_plot(
        relative_time,
        reduced_states,
        model_reduced_trajectories[
            :, :coordinate_rom_trajectory_count, :
        ],
        coordinate_modes,
        output_dir / "coordinate_ensemble_trajectories.png",
        args.dpi,
    )
    save_experimental_mean_plot(
        relative_time,
        x,
        experimental_full_mean,
        output_dir / "experimental_ensemble_mean.png",
        args.dpi,
    )
    save_regularization_plot(
        candidates,
        mean_errors,
        covariance_errors,
        best_mean_index,
        regularized_operator,
        output_dir / "regularization_errors.png",
        args.dpi,
    )
    save_relative_error_plot(
        relative_time,
        mean_error_time,
        covariance_error_time,
        output_dir / "relative_mean_covariance_errors.png",
        args.dpi,
    )
    save_reduced_mean_plot(
        relative_time,
        experimental_reduced_mean,
        model_reduced_mean,
        output_dir / "reduced_mean_comparison.png",
        args.dpi,
    )
    save_reduced_covariance_plot(
        relative_time,
        experimental_reduced_covariance,
        model_reduced_covariance,
        output_dir / "reduced_covariance_comparison.png",
        args.dpi,
    )
    save_full_mean_comparison_plot(
        relative_time,
        x,
        experimental_full_mean,
        model_full_mean,
        output_dir / "full_mean_comparison.png",
        args.dpi,
    )

    print(f"Saved all results to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
