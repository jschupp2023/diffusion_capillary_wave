"""Train a stochastic ROM on half the realizations and test on the other half.

The POD basis and stochastic operators are learned only from the training half.
The supplied rank is used directly; there is no rank sweep.  Reduced-space
means and covariances of the held-out data are compared with ROM predictions.

Example
-------
    python evaluate_unseen_data.py 0p04 --rank 23 --model-form A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import helpers as sto_opinf


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
    parser = argparse.ArgumentParser(
        description=(
            "Fit a fixed-rank stochastic OpInf model to a 50% training split "
            "and evaluate reduced mean/covariance on the unseen 50%."
        )
    )
    parser.add_argument("power", choices=POWER_LABELS, help="Power label, e.g. 0p04.")
    parser.add_argument("--rank", type=int, required=True, help="Fixed POD rank r.")
    parser.add_argument(
        "--model-form",
        type=str.upper,
        choices=("A", "AB", "AN", "ABN"),
        default="A",
        help=(
            "Drift terms to infer: A (linear, default), AB, AN, or ABN. "
            "Here B is the additive-input term and N is the bilinear term."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
        help="Directory containing one subdirectory per power.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output folder (default: "
            "unseen_data_results/<power>_r<rank>_<model-form>)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Split and ROM simulation seed (default: 42)."
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI (default: 200).")
    return parser.parse_args()


def reduced_statistics(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and pagewise sample covariance."""
    mean = states.mean(axis=1)
    covariance = sto_opinf.page_cov(states, transpose_pages=True)
    return mean, covariance


def relative_errors(
    data_mean: np.ndarray,
    data_covariance: np.ndarray,
    model_mean: np.ndarray,
    model_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the notebook's relative errors at every time point."""
    eps = np.finfo(float).eps
    mean_numerator = np.linalg.norm(data_mean - model_mean, axis=0)
    mean_denominator = np.maximum(np.linalg.norm(data_mean, axis=0), eps)

    covariance_numerator = np.linalg.norm(
        data_covariance - model_covariance, axis=(0, 1)
    )
    covariance_denominator = np.maximum(
        np.linalg.norm(data_covariance, axis=(0, 1)), eps
    )
    return mean_numerator / mean_denominator, covariance_numerator / covariance_denominator


def save_error_plot(
    time: np.ndarray,
    mean_error: np.ndarray,
    covariance_error: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, values, title, color in zip(
        axes,
        (mean_error, covariance_error),
        ("Reduced mean error", "Reduced covariance error"),
        ("royalblue", "tomato"),
    ):
        ax.semilogy(time, np.maximum(values, np.finfo(float).tiny), color=color, lw=1.5)
        ax.set(title=title, xlabel="Time [s]", ylabel="Relative error")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Stochastic OpInf prediction on unseen data")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_mean_plot(
    time: np.ndarray,
    data_mean: np.ndarray,
    model_mean: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    difference = np.abs(data_mean - model_mean)
    extent = (float(time[0]), float(time[-1]), 1, data_mean.shape[0])
    value_limit = max(float(np.max(np.abs(data_mean))), float(np.max(np.abs(model_mean))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    image = None
    for ax, values, title in zip(
        axes[:2], (data_mean, model_mean), ("Held-out data", "ROM")
    ):
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-value_limit,
            vmax=value_limit,
        )
        ax.set_title(title)
    assert image is not None
    fig.colorbar(image, ax=axes[:2], label="Reduced mean coefficient")

    error_image = axes[2].imshow(
        difference, aspect="auto", origin="lower", extent=extent, cmap="magma"
    )
    axes[2].set_title("Absolute difference")
    fig.colorbar(error_image, ax=axes[2], label="Absolute error")
    for ax in axes:
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("POD mode")
    fig.suptitle("Reduced mean on unseen data")
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.14, top=0.82, wspace=0.22)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_covariance_plot(
    time: np.ndarray,
    data_covariance: np.ndarray,
    model_covariance: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    indices = np.unique(np.linspace(0, len(time) - 1, 3, dtype=int))
    fig, axes = plt.subplots(
        3, len(indices), figsize=(4 * len(indices), 10), squeeze=False
    )

    for column, index in enumerate(indices):
        limit = max(
            float(np.max(np.abs(data_covariance[:, :, index]))),
            float(np.max(np.abs(model_covariance[:, :, index]))),
            np.finfo(float).eps,
        )
        for row, covariance in enumerate((data_covariance, model_covariance)):
            image = axes[row, column].imshow(
                covariance[:, :, index],
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)

        difference = np.abs(data_covariance[:, :, index] - model_covariance[:, :, index])
        image = axes[2, column].imshow(difference, origin="lower", cmap="magma")
        fig.colorbar(image, ax=axes[2, column], fraction=0.046, pad=0.04)
        axes[0, column].set_title(f"t = {time[index]:.4g} s")

    for row, label in enumerate(("Held-out data", "ROM", "Absolute difference")):
        axes[row, 0].set_ylabel(f"{label}\nPOD mode")
    for ax in axes.flat:
        ax.set_xlabel("POD mode")
    fig.suptitle("Reduced covariance on unseen data")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.rank < 2:
        raise ValueError("--rank must be at least 2 for covariance evaluation.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")

    output_dir = args.output_dir or Path("unseen_data_results") / (
        f"{args.power}_r{args.rank}_{args.model_form}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and preprocessing {args.power} data...")
    dataset = sto_opinf.QDataset.load(
        args.power, POWER_LABELS[args.power], str(args.data_dir)
    ).preprocess()
    states_all = dataset.Qstate_all
    time = dataset.tt
    if states_all is None or time is None:
        raise RuntimeError("Dataset preprocessing did not produce states and time.")

    # QDataset retains both the raw experiments and the concatenated array.
    # They are no longer needed after preprocessing and are large enough to
    # matter for the subsequent train/test copies.
    dataset.Q.clear()
    if dataset.Q_split is not None:
        dataset.Q_split.clear()
    dataset.Qstate_all = None

    n_space, n_realizations, n_time = states_all.shape
    if args.rank > n_space:
        raise ValueError(f"--rank cannot exceed the spatial dimension ({n_space}).")
    n_test = n_realizations // 2
    if n_test < 2 or n_realizations - n_test < 2:
        raise ValueError("The 50% split needs at least two train and two test realizations.")

    # Match the notebook's seeded, random 50/50 split.
    rng = np.random.RandomState(args.seed)
    test_indices = np.sort(rng.choice(n_realizations, n_test, replace=False))
    train_mask = np.ones(n_realizations, dtype=bool)
    train_mask[test_indices] = False
    train_indices = np.flatnonzero(train_mask)
    train_states = states_all[:, train_indices, :]
    test_states = states_all[:, test_indices, :]
    del states_all, dataset

    print(f"Computing the rank-{args.rank} training-only POD basis...")
    train_snapshots = train_states.reshape(n_space, -1)
    gram = train_snapshots @ train_snapshots.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[: args.rank]]

    reduced_train = (basis.T @ train_snapshots).reshape(
        args.rank, len(train_indices), n_time
    )
    reduced_test = (basis.T @ test_states.reshape(n_space, -1)).reshape(
        args.rank, len(test_indices), n_time
    )
    del train_snapshots, train_states, test_states, gram, eigenvalues, eigenvectors
    train_mean, train_covariance = reduced_statistics(reduced_train)
    test_mean, test_covariance = reduced_statistics(reduced_test)

    is_bilinear = "N" in args.model_form
    is_bu = "B" in args.model_form

    # An A-only model does not use u at all. Keep it identically zero so the
    # saved configuration and simulation make that choice explicit.
    if is_bilinear or is_bu:
        input_signal = np.cos(2 * np.pi * 7e6 * time)
    else:
        input_signal = np.zeros_like(time)

    time_step = float(time[1] - time[0])
    h_regularization = 5e4
    b_regularization = 0.0

    # Match the notebook's grids: regularize N when it is present; otherwise
    # regularize A. RegularizationGrid calls this its N_reg axis internally,
    # but it is simply the outer candidate axis used by run_grid_search().
    if is_bilinear:
        drift_regularization = np.logspace(8, 10, 10)
        regularized_operator = "N"
    else:
        drift_regularization = np.logspace(-1, 5, 10)
        regularized_operator = "A"

    def get_lambda(i: int, j: int) -> np.ndarray:
        candidate = drift_regularization[i]
        if is_bilinear and is_bu:
            return np.array([0.0, b_regularization, candidate])
        if is_bilinear:
            return np.array([0.0, candidate])
        if is_bu:
            return np.array([candidate, b_regularization])
        return np.array([candidate])

    training_data = sto_opinf.StochasticOpInfTrainingData(
        train_mean,
        train_covariance,
        reduced_train,
        input_signal,
        basis,
        time_step,
    )
    grid = sto_opinf.RegularizationGrid(
        drift_regularization, h_regularization, np.arange(1), get_lambda, args.seed
    )

    print(f"Fitting {args.model_form} regularization candidates on training data...")
    grid_results = sto_opinf.run_grid_search(
        training_data, grid, is_bilinear, is_bu, sigma=1.0
    )
    training_errors = np.asarray(grid_results["E_error"], dtype=float)
    best_i, best_j = np.unravel_index(
        np.nanargmin(training_errors), training_errors.shape
    )
    operators = grid_results["ROMs_all"][best_i][best_j]

    model = sto_opinf.StochasticOpInfROM(isbilinear=is_bilinear, isBu=is_bu)
    for name in ("Ehat", "Ahat", "Bhat", "Nhat", "Mhat", "Khat"):
        setattr(model, name, operators[name])

    print("Simulating the selected model from held-out initial conditions...")
    model_mean, model_covariance, _ = model.simulate(
        reduced_test[:, :, 0],
        input_signal,
        time_step,
        sigma=1.0,
        batch_size=len(test_indices),
        L=len(test_indices),
        seed=args.seed,
    )

    mean_error, covariance_error = relative_errors(
        test_mean, test_covariance, model_mean, model_covariance
    )
    aggregate_mean_error = np.linalg.norm(test_mean - model_mean) / np.linalg.norm(
        test_mean
    )
    aggregate_covariance_error = sto_opinf.page_norm(
        test_covariance - model_covariance
    ) / sto_opinf.page_norm(test_covariance)

    metrics = {
        "power": args.power,
        "rank": args.rank,
        "model_form": args.model_form,
        "isbilinear": is_bilinear,
        "isBu": is_bu,
        "seed": args.seed,
        "train_fraction": 0.5,
        "train_realizations": len(train_indices),
        "test_realizations": len(test_indices),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "regularized_operator": regularized_operator,
        "selected_drift_regularization": float(drift_regularization[best_i]),
        "training_mean_error": float(grid_results["E_error"][best_i, best_j]),
        "training_covariance_error": float(grid_results["C_error"][best_i, best_j]),
        "test_mean_error": float(aggregate_mean_error),
        "test_covariance_error": float(aggregate_covariance_error),
        "mean_time_resolved_test_error": float(mean_error.mean()),
        "mean_time_resolved_covariance_test_error": float(covariance_error.mean()),
        "maximum_mean_test_error": float(mean_error.max()),
        "maximum_covariance_test_error": float(covariance_error.max()),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "reduced_statistics.npz",
        time=time,
        basis=basis,
        test_mean=test_mean,
        model_mean=model_mean,
        test_covariance=test_covariance,
        model_covariance=model_covariance,
        mean_error=mean_error,
        covariance_error=covariance_error,
    )
    save_error_plot(
        time, mean_error, covariance_error, output_dir / "relative_errors.png", args.dpi
    )
    save_mean_plot(
        time, test_mean, model_mean, output_dir / "reduced_mean.png", args.dpi
    )
    save_covariance_plot(
        time,
        test_covariance,
        model_covariance,
        output_dir / "reduced_covariance.png",
        args.dpi,
    )

    print(f"Test mean error:       {aggregate_mean_error:.4e}")
    print(f"Test covariance error: {aggregate_covariance_error:.4e}")
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
