"""Create compact single-column paper figures from saved comparison data.

This script only reads existing NPZ files under ``outputs/comparison_runs``.
It generates:

1. A 4x1 final-performance summary over 20/40/60/80/100 views.
2. A 3x1 representative convergence figure for a selected view count.

No model loading or reconstruction is performed. The original single-scale
DiffPIR implementation is displayed simply as ``DiffPIR``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


VIEWS = (20, 40, 60, 80, 100)
METHODS = ("FBP", "DDS", "DiffPIR", "Multiscale DiffPIR")
METRIC_ARCHIVE_KEYS = {
    "FBP": "fbp",
    "DDS": "dds",
    "DiffPIR": "single_scale",
    "Multiscale DiffPIR": "multi_scale",
}
TIME_ARCHIVE_KEYS = {
    "FBP": "fbp_ms",
    "DDS": "dds_ms",
    "DiffPIR": "original_ms",
    "Multiscale DiffPIR": "multiscale_ms",
}
STYLES = {
    "FBP": {"color": "#2ca02c", "marker": "o", "linestyle": ":"},
    "DDS": {"color": "#9467bd", "marker": "s", "linestyle": "-."},
    "DiffPIR": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "Multiscale DiffPIR": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "--",
    },
}
METRICS = {
    "psnr": {"label": "PSNR (dB) ↑", "panel": "(a)"},
    "ssim": {"label": "SSIM ↑", "panel": "(b)"},
    "lpips": {"label": "LPIPS ↓", "panel": "(c)"},
    "time": {"label": "Runtime (s/image) ↓", "panel": "(d)"},
}


def _final_track_values(tracks: np.ndarray) -> np.ndarray:
    """Return each sample's value at the final (minimum) saved timestep."""
    return np.asarray(
        [float(track[min(track)]) for track in tracks], dtype=np.float64
    )


def load_final_statistics(
    outputs_dir: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    """Load 500-sample means and standard errors for every metric and method."""
    means = {metric: {method: [] for method in METHODS} for metric in METRICS}
    standard_errors = {
        metric: {method: [] for method in METHODS} for metric in METRICS
    }

    for view in VIEWS:
        evaluation_dir = (
            outputs_dir / "comparison_runs" / f"nview_{view}_evals_500"
        )
        timing_path = (
            outputs_dir
            / "comparison_runs"
            / f"nview_{view}_time_500"
            / "time"
            / "time_per_image.npz"
        )
        if not timing_path.is_file():
            raise FileNotFoundError(f"Missing timing data: {timing_path}")

        for metric in ("psnr", "ssim", "lpips"):
            metric_path = evaluation_dir / metric / f"{metric}_raw_curves.npz"
            if not metric_path.is_file():
                raise FileNotFoundError(f"Missing metric data: {metric_path}")
            with np.load(metric_path, allow_pickle=True) as archive:
                for method in METHODS:
                    key = METRIC_ARCHIVE_KEYS[method]
                    raw_values = archive[key]
                    values = (
                        np.asarray(raw_values, dtype=np.float64)
                        if key == "fbp"
                        else _final_track_values(raw_values)
                    )
                    if len(values) != 500:
                        raise ValueError(
                            f"Expected 500 values in {metric_path} for {method}; "
                            f"found {len(values)}."
                        )
                    means[metric][method].append(float(values.mean()))
                    standard_errors[metric][method].append(
                        float(values.std(ddof=1) / np.sqrt(len(values)))
                    )

        with np.load(timing_path, allow_pickle=False) as timing:
            for method in METHODS:
                values = np.asarray(
                    timing[TIME_ARCHIVE_KEYS[method]], dtype=np.float64
                ) / 1000.0
                if len(values) != 500:
                    raise ValueError(
                        f"Expected 500 values in {timing_path} for {method}; "
                        f"found {len(values)}."
                    )
                means["time"][method].append(float(values.mean()))
                standard_errors["time"][method].append(
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                )

    for metric in METRICS:
        for method in METHODS:
            means[metric][method] = np.asarray(
                means[metric][method], dtype=np.float64
            )
            standard_errors[metric][method] = np.asarray(
                standard_errors[metric][method], dtype=np.float64
            )
    return means, standard_errors


def _apply_paper_style(axis: plt.Axes) -> None:
    axis.grid(True, color="#d5d5d5", linestyle=":", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5, direction="out")


def plot_final_performance(
    means: dict[str, dict[str, np.ndarray]],
    standard_errors: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    dpi: int,
    legend_y: float = 0.94,
) -> Figure:
    """Plot final average quality and runtime as a compact 2x2 figure."""
    figure, axes = plt.subplots(2, 2, figsize=(6.7, 5.0))

    metric_order = [
        ("psnr", axes[0, 0]),
        ("ssim", axes[0, 1]),
        ("lpips", axes[1, 0]),
        ("time", axes[1, 1]),
    ]

    for metric, axis in metric_order:
        settings = METRICS[metric]
        for method in METHODS:
            style = STYLES[method]
            axis.errorbar(
                VIEWS,
                means[metric][method],
                yerr=standard_errors[metric][method],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=2.2,
                elinewidth=0.8,
            )
        axis.set_ylabel(settings["label"], fontsize=9.5)
        axis.set_xticks(VIEWS)
        axis.set_xticklabels(VIEWS, fontsize=8.5)
        _apply_paper_style(axis)

    axes[1, 0].set_xlabel("Number of projection views", fontsize=9.5)
    axes[1, 1].set_xlabel("Number of projection views", fontsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=5.0,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    figure.subplots_adjust(
        left=0.108, right=0.98, bottom=0.12, top=0.93, hspace=0.42, wspace=0.32
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    return figure


def load_average_convergence(
    outputs_dir: Path, view: int
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray] | float]]:
    """Load average convergence curves for one representative view count."""
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | float]] = {}
    evaluation_dir = outputs_dir / "comparison_runs" / f"nview_{view}_evals_500"

    for metric in ("psnr", "ssim", "lpips"):
        metric_path = evaluation_dir / metric / f"{metric}_raw_curves.npz"
        if not metric_path.is_file():
            raise FileNotFoundError(f"Missing metric data: {metric_path}")
        result[metric] = {}
        with np.load(metric_path, allow_pickle=True) as archive:
            for method in METHODS:
                key = METRIC_ARCHIVE_KEYS[method]
                values = archive[key]
                if len(values) != 500:
                    raise ValueError(
                        f"Expected 500 values in {metric_path} for {method}; "
                        f"found {len(values)}."
                    )
                if method == "FBP":
                    result[metric][method] = float(
                        np.asarray(values, dtype=np.float64).mean()
                    )
                else:
                    timesteps = np.asarray(
                        sorted(values[0].keys(), reverse=True), dtype=np.int64
                    )
                    averages = np.asarray(
                        [
                            np.mean([float(track[timestep]) for track in values])
                            for timestep in timesteps
                        ],
                        dtype=np.float64,
                    )
                    result[metric][method] = (timesteps, averages)
    return result


def plot_representative_convergence(
    convergence: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray] | float]
    ],
    output_path: Path,
    *,
    view: int,
    dpi: int,
    legend_y: float = 0.94,
) -> Figure:
    """Plot PSNR/SSIM/LPIPS convergence for one representative view count."""
    figure, axes = plt.subplots(3, 1, figsize=(6.7, 5.5))

    for i, (axis, metric) in enumerate(
        zip(axes, ("psnr", "ssim", "lpips"), strict=True)
    ):
        settings = METRICS[metric]
        for method in METHODS:
            style = STYLES[method]
            values = convergence[metric][method]
            if method == "FBP":
                axis.axhline(
                    float(values),
                    label=method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.7,
                )
            else:
                timesteps, averages = values
                axis.plot(
                    timesteps,
                    averages,
                    label=method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.7,
                )
        axis.invert_xaxis()
        axis.set_ylabel(settings["label"], fontsize=9.5)
        axis.tick_params(labelsize=8.5)
        if i < 2:
            axis.set_xticklabels([])
            axis.set_xlabel("")
        _apply_paper_style(axis)

    axes[-1].set_xlabel("Diffusion timestep $t$", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=5.0,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    figure.subplots_adjust(
        left=0.13, right=0.98, bottom=0.09, top=0.93, hspace=0.08
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact single-column paper metric figures."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing comparison_runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_metric_summary"),
    )
    parser.add_argument(
        "--representative-view",
        type=int,
        choices=VIEWS,
        default=60,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.94,
        help="Y position of legend bbox_to_anchor (figure coords). Lower=closer to plots.",
    )
    args = parser.parse_args()

    means, standard_errors = load_final_statistics(args.outputs_dir)
    final_output = args.output_dir / "final_metrics_vs_views.png"
    final_figure = plot_final_performance(
        means, standard_errors, final_output, dpi=args.dpi, legend_y=args.legend_y
    )
    plt.close(final_figure)

    convergence = load_average_convergence(
        args.outputs_dir, args.representative_view
    )
    convergence_output = (
        args.output_dir
        / f"convergence_{args.representative_view}_views.png"
    )
    convergence_figure = plot_representative_convergence(
        convergence,
        convergence_output,
        view=args.representative_view,
        dpi=args.dpi,
        legend_y=args.legend_y,
    )
    plt.close(convergence_figure)

    print("Loaded exact statistics from 500 saved experimental samples.")
    print(f"Saved final-performance summary: {final_output.resolve()}")
    print(f"Saved representative convergence: {convergence_output.resolve()}")


if __name__ == "__main__":
    main()
