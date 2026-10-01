"""Shared Matplotlib helpers for fixed-camera capillary surface plots."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np


@dataclass(frozen=True)
class SurfaceFigure:
    figure: Figure
    canvas: FigureCanvasAgg
    axes: object
    norm: Normalize
    cmap: object


def padded_limits(
    field: np.ndarray, padding_fraction: float = 0.08
) -> tuple[float, float]:
    """Return finite plotting limits with a small margin around the data."""
    values = np.asarray(field)
    if not np.isfinite(values).all():
        raise ValueError("Surface contains NaN or infinite values.")
    zmin, zmax = float(values.min()), float(values.max())
    if zmin == zmax:
        zmin, zmax = zmin - 1.0, zmax + 1.0
    padding = padding_fraction * (zmax - zmin)
    return zmin - padding, zmax + padding


def make_surface_figure(
    x: np.ndarray,
    y: np.ndarray,
    zlim: tuple[float, float],
    *,
    x_unit: str,
    y_unit: str,
    z_unit: str,
    figsize: tuple[float, float],
    dpi: float,
    elev: float = 30.0,
    azim: float = -60.0,
) -> SurfaceFigure:
    """Build the common fixed-camera 3-D surface figure and color scale."""
    zmin, zmax = zlim
    if not np.isfinite((zmin, zmax)).all() or zmin >= zmax:
        raise ValueError("z limits must be finite and strictly increasing.")
    figure = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.02, 0.04, 0.76, 0.88), projection="3d")
    axes.view_init(elev=elev, azim=azim)
    axes.set_box_aspect((1, 1, 0.48))
    axes.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(zmin, zmax))
    axes.set_xlabel(f"x [{x_unit}]", labelpad=2)
    axes.set_ylabel(f"y [{y_unit}]", labelpad=2)
    axes.set_zlabel(f"height [{z_unit}]", labelpad=2)
    axes.tick_params(labelsize=7, pad=0)
    norm = Normalize(vmin=zmin, vmax=zmax)
    cmap = matplotlib.colormaps["viridis"]
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=figure.add_axes((0.87, 0.2, 0.025, 0.58)),
    )
    colorbar.set_label(f"height [{z_unit}]", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return SurfaceFigure(figure, canvas, axes, norm, cmap)


def draw_surface(
    plot: SurfaceFigure,
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    full_resolution: bool = False,
):
    """Draw one field, optionally forcing every stored grid point to be used."""
    x_grid, y_grid = np.meshgrid(x, y)
    resolution = {}
    if full_resolution:
        resolution = {"rcount": field.shape[0], "ccount": field.shape[1]}
    return plot.axes.plot_surface(
        x_grid,
        y_grid,
        field,
        facecolors=plot.cmap(plot.norm(field)),
        linewidth=0,
        antialiased=False,
        shade=False,
        **resolution,
    )
