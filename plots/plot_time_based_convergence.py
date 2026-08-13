"""Plot quality metrics versus wall-clock time instead of diffusion timestep.

Each diffusion method progresses through timesteps at a different per-step cost.
This script uses total per-image runtime divided by number of timesteps to
estimate cumulative wall-clock time, then plots PSNR/SSIM/LPIPS against it.

FBP (non-iterative) appears as a horizontal reference line at its total time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


VIEWS = (20, 40, 60, 80, 100)
METHODS = ("DDS", "DiffPIR", "Multiscale DiffPIR")

METRIC_ARCHIVE_KEYS = {
    "DDS": "dds",
    "DiffPIR": "single_scale",
    "Multiscale DiffPIR": "multi_scale",
}
TIME_ARCHIVE_KEYS = {
    "DDS": "dds_ms",
    "DiffPIR": "original_ms",
    "Multiscale DiffPIR": "multiscale_ms",
}
STYLES = {
    "DDS": {"color": "#9467bd", "marker": "s", "linestyle": "-."},
    "DiffPIR": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "Multiscale DiffPIR": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "--",
    },
    "FBP": {"color": "#2ca02c", "linestyle": ":"},
}
METRICS = {
    "psnr": {"label": "PSNR (dB) ↑", "panel": "(a)"},
    "ssim": {"label": "SSIM ↑", "panel": "(b)"},
    "lpips": {"label": "LPIPS ↓", "panel": "(c)"},
}


def load_time_based_convergence(
    outputs_dir: Path, view: int
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray] | tuple[float, float]]]:
    """Load convergence curves and map timesteps to cumulative wall-clock time.

    Returns
    -------
    result : dict
        {metric: {method: (cumulative_time_seconds, metric_values)}}
        FBP → (total_time, fbp_value) — single point.
    """
    evaluation_dir = outputs_dir / "comparison_runs" / f"nview_{view}_evals_500"
    timing_path = (
        outputs_dir
        / "comparison_runs"
        / f"nview_{view}_time_500"
        / "time"
        / "time_per_image.npz"
    )

    with np.load(timing_path, allow_pickle=False) as timing:
        per_image_times = {
            method: np.asarray(timing[TIME_ARCHIVE_KEYS[method]], dtype=np.float64)
            / 1000.0
            for method in METHODS
        }
        fbp_total = float(
            np.asarray(timing["fbp_ms"], dtype=np.float64).mean() / 1000.0
        )

    result: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray] | tuple[float, float]]
    ] = {}

    for metric in ("psnr", "ssim", "lpips"):
        metric_path = evaluation_dir / metric / f"{metric}_raw_curves.npz"
        result[metric] = {}

        with np.load(metric_path, allow_pickle=True) as archive:
            # FBP: single point
            fbp_values = np.asarray(archive["fbp"], dtype=np.float64)
            result[metric]["FBP"] = (fbp_total, float(fbp_values.mean()))

            for method in METHODS:
                key = METRIC_ARCHIVE_KEYS[method]
                tracks = archive[key]
                num_samples = len(tracks)
                num_recorded_steps = len(tracks[0])

                # Average total wall-clock time for this method
                avg_total_time = float(per_image_times[method].mean())

                is_multiscale = method == "Multiscale DiffPIR"

                # Timesteps (descending: T, T-Δ, …, 0)
                timesteps = np.asarray(
                    sorted(tracks[0].keys(), reverse=True), dtype=np.int64
                )

                # Average metric at each timestep across 500 samples
                metric_means = np.asarray(
                    [
                        np.mean([float(track[timestep]) for track in tracks])
                        for timestep in timesteps
                    ],
                    dtype=np.float64,
                )

                if is_multiscale:
                    # Estimate full-res per-step from DiffPIR's known cost.
                    # DDS/DiffPIR: 50 DDIM steps → 47 unique rounded timesteps.
                    # Multiscale warmup = total − num_recorded_steps × per_step.
                    diffpir_total = float(per_image_times["DiffPIR"].mean())
                    full_res_per_step = diffpir_total / 50.0  # DiffPIR: 50 DDIM full-res steps
                    warmup = avg_total_time - num_recorded_steps * full_res_per_step
                    cumulative_time = warmup + np.arange(
                        1, num_recorded_steps + 1, dtype=np.float64
                    ) * full_res_per_step
                else:
                    # 50 DDIM steps produce 47 unique timesteps after rounding;
                    # divide by 50 for true per-iteration cost.
                    time_per_step = avg_total_time / 50.0
                    cumulative_time = (
                        np.arange(1, num_recorded_steps + 1, dtype=np.float64)
                        * time_per_step
                    )
                    # Prepend (0, first_value) — valid since all steps full-resolution
                    cumulative_time = np.concatenate([[0.0], cumulative_time])
                    metric_means = np.concatenate([[metric_means[0]], metric_means])

                result[metric][method] = (cumulative_time, metric_means)

    return result


def _apply_paper_style(axis: plt.Axes) -> None:
    axis.grid(True, color="#d5d5d5", linestyle=":", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5, direction="out")


def plot_time_convergence(
    convergence: dict,
    output_path: Path,
    *,
    view: int,
    dpi: int,
    legend_y: float = 0.94,
    max_time: float | None = None,
) -> Figure:
    """3x1 figure: PSNR/SSIM/LPIPS vs cumulative wall-clock time."""
    figure, axes = plt.subplots(3, 1, figsize=(6.7, 5.5))

    for i, (axis, metric) in enumerate(
        zip(axes, ("psnr", "ssim", "lpips"), strict=True)
    ):
        settings = METRICS[metric]
        data = convergence[metric]

        # FBP: horizontal reference line (instantaneous, no time axis)
        _, fbp_val = data["FBP"]
        axis.axhline(
            fbp_val,
            label="FBP",
            color=STYLES["FBP"]["color"],
            linestyle=STYLES["FBP"]["linestyle"],
            linewidth=1.4,
            alpha=0.7,
            zorder=0,
        )

        # Diffusion convergence curves with light fill toward final value
        for method in METHODS:
            style = STYLES[method]
            times, values = data[method]
            axis.plot(
                times,
                values,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                zorder=2,
            )
            # Subtle filled area under curve
            axis.fill_between(
                times,
                values,
                values[-1],
                color=style["color"],
                alpha=0.06,
                zorder=1,
            )

        axis.set_ylabel(settings["label"], fontsize=9.5)
        axis.tick_params(labelsize=8.5)
        if i < 2:
            axis.set_xticklabels([])
            axis.set_xlabel("")
        axis.set_xlim(left=0)
        if max_time is not None:
            axis.set_xlim(0, max_time)
        _apply_paper_style(axis)

    axes[-1].set_xlabel("Wall-clock time (s)", fontsize=9.5)

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
        description="Plot quality vs. wall-clock time convergence."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/time_based/"),
    )
    parser.add_argument(
        "--view",
        type=int,
        choices=VIEWS,
        default=60,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.94,
        help="Legend Y position in figure coords.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Clip x-axis to this max time (seconds). If not set, auto-scale.",
    )
    args = parser.parse_args()

    convergence = load_time_based_convergence(args.outputs_dir, args.view)

    output_path = args.output_dir / f"time_convergence_{args.view}_views.png"
    figure = plot_time_convergence(
        convergence,
        output_path,
        view=args.view,
        dpi=args.dpi,
        legend_y=args.legend_y,
        max_time=args.max_time,
    )
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
