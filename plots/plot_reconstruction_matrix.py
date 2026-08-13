"""Create a qualitative CT reconstruction comparison matrix.

The plotting function consumes NumPy arrays and does not depend on the model or
data-loading code in this repository. Images should represent the same CT slice
for every view count and method.

Expected input layout::

    images_dict = {
        20: {
            "FBP": fbp_20,
            "DDS": dds_20,
            "Single-scale DiffPIR": single_20,
            "Multiscale DiffPIR": multi_20,
        },
        # 40, 60, 80, and 100 use the same method keys.
    }
    gt_images = {20: gt, 40: gt, 60: gt, 80: gt, 100: gt}
    roi_coords = {view: (x, y, width, height) for view in images_dict}

Metrics can be supplied manually or loaded from the comparison runs with
``load_metrics_from_outputs``. The latter reads the final diffusion timestep
for PSNR/SSIM/LPIPS and the separate timing runs for runtime.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


VIEW_LEVELS = (20, 40, 60, 80, 100)
METHODS = (
    "FBP",
    "DDS",
    "Single-scale DiffPIR",
    "Multiscale DiffPIR",
    "GT",
)

_METHOD_DISPLAY_NAMES = {
    "Single-scale DiffPIR": "DiffPIR",
}

_METHOD_TO_ARCHIVE_KEY = {
    "FBP": "fbp",
    "DDS": "dds",
    "Single-scale DiffPIR": "single_scale",
    "Multiscale DiffPIR": "multi_scale",
}

_METHOD_TO_TIME_KEY = {
    "FBP": "fbp_ms",
    "DDS": "dds_ms",
    "Single-scale DiffPIR": "original_ms",
    "Multiscale DiffPIR": "multiscale_ms",
}

ATTENUATION_MAX = 0.04825
DEFAULT_SEED = 97


def _as_2d_float(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{label} must become 2D after squeeze; got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return array


def _validate_roi(
    roi: Sequence[int], image_shape: tuple[int, int], view: int
) -> tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError(f"ROI for {view} views must be (x, y, width, height).")
    x, y, width, height = map(int, roi)
    image_height, image_width = image_shape
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"ROI for {view} views has invalid values: {roi}.")
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {roi} exceeds image bounds {(image_width, image_height)} "
            f"for {view} views."
        )
    return x, y, width, height


def _resize_roi(roi: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_CUBIC if max(roi.shape) < size else cv2.INTER_AREA
    return cv2.resize(roi, (size, size), interpolation=interpolation)


def select_detail_roi(
    image: np.ndarray,
    *,
    roi_size: int = 96,
    border_fraction: float = 0.12,
    excluded_rois: Sequence[Sequence[int]] = (),
) -> tuple[int, int, int, int]:
    """Select a square, high-detail ROI from the central anatomy."""
    image = _as_2d_float(image, "GT image")
    height, width = image.shape
    if roi_size <= 0 or roi_size > min(height, width):
        raise ValueError(f"roi_size={roi_size} is invalid for image shape {image.shape}.")

    normalized = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    detail = cv2.magnitude(grad_x, grad_y)
    local_detail = cv2.boxFilter(
        detail,
        ddepth=-1,
        ksize=(roi_size, roi_size),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT,
    )

    half = roi_size // 2
    margin_x = max(half, int(round(width * border_fraction)))
    margin_y = max(half, int(round(height * border_fraction)))
    valid = np.zeros_like(local_detail, dtype=bool)
    valid[margin_y : height - margin_y, margin_x : width - margin_x] = True
    for excluded_roi in excluded_rois:
        excluded_x, excluded_y, excluded_width, excluded_height = map(
            int, excluded_roi
        )
        excluded_center_x = excluded_x + excluded_width // 2
        excluded_center_y = excluded_y + excluded_height // 2
        valid[
            max(0, excluded_center_y - roi_size) : min(
                height, excluded_center_y + roi_size + 1
            ),
            max(0, excluded_center_x - roi_size) : min(
                width, excluded_center_x + roi_size + 1
            ),
        ] = False
    local_detail[~valid] = -np.inf

    center_y, center_x = np.unravel_index(np.argmax(local_detail), image.shape)
    x = int(np.clip(center_x - half, 0, width - roi_size))
    y = int(np.clip(center_y - half, 0, height - roi_size))
    return x, y, roi_size, roi_size


def _final_curve_value(curves: np.ndarray, sample_index: int) -> float:
    curve = curves[sample_index]
    if not isinstance(curve, Mapping) or not curve:
        raise ValueError("Expected each saved metric curve to be a non-empty mapping.")
    final_timestep = min(int(timestep) for timestep in curve)
    return float(curve[final_timestep])


def load_metrics_from_outputs(
    outputs_dir: str | Path,
    sample_index: int,
    *,
    view_levels: Sequence[int] = VIEW_LEVELS,
    evaluation_run_pattern: str = "nview_{view}_evals_500",
    timing_run_pattern: str = "nview_{view}_time_500",
    average_runtime: bool = True,
) -> dict[int, dict[str, dict[str, float]]]:
    """Load one sample's final metrics and runtime from ``outputs``.

    ``sample_index`` is zero-based. Runtime defaults to the mean over the
    separate 500-image timing run because this is usually more stable for a
    paper figure. Set ``average_runtime=False`` to use the matching array index.
    NPZ object arrays are loaded with pickle because Comparison.py stores each
    convergence curve as a Python dictionary.
    """
    outputs_dir = Path(outputs_dir)
    runs_dir = outputs_dir / "comparison_runs"
    result: dict[int, dict[str, dict[str, float]]] = {}

    for view in view_levels:
        evaluation_dir = runs_dir / evaluation_run_pattern.format(view=view)
        timing_dir = runs_dir / timing_run_pattern.format(view=view)
        metric_archives: dict[str, dict[str, np.ndarray]] = {}

        for metric_name in ("psnr", "ssim", "lpips"):
            archive_path = evaluation_dir / metric_name / f"{metric_name}_raw_curves.npz"
            if not archive_path.is_file():
                raise FileNotFoundError(f"Missing metric archive: {archive_path}")
            with np.load(archive_path, allow_pickle=True) as archive:
                metric_archives[metric_name] = {
                    key: archive[key].copy() for key in archive.files
                }

        time_path = timing_dir / "time" / "time_per_image.npz"
        if not time_path.is_file():
            raise FileNotFoundError(f"Missing timing archive: {time_path}")
        with np.load(time_path) as archive:
            timing = {key: archive[key].copy() for key in archive.files}

        sample_count = len(metric_archives["psnr"]["fbp"])
        if not 0 <= sample_index < sample_count:
            raise IndexError(
                f"sample_index={sample_index} is outside [0, {sample_count - 1}] "
                f"for {view} views."
            )

        result[view] = {}
        for method in METHODS[:-1]:
            archive_key = _METHOD_TO_ARCHIVE_KEY[method]
            time_key = _METHOD_TO_TIME_KEY[method]
            method_metrics: dict[str, float] = {}

            for metric_name in ("psnr", "ssim", "lpips"):
                values = metric_archives[metric_name][archive_key]
                if archive_key == "fbp":
                    method_metrics[metric_name] = float(values[sample_index])
                else:
                    method_metrics[metric_name] = _final_curve_value(
                        values, sample_index
                    )

            runtime_values = np.asarray(timing[time_key], dtype=np.float64)
            if average_runtime:
                method_metrics["runtime_ms"] = float(runtime_values.mean())
            else:
                if sample_index >= len(runtime_values):
                    raise IndexError(
                        f"Timing data for {method}, {view} views has only "
                        f"{len(runtime_values)} samples."
                    )
                method_metrics["runtime_ms"] = float(runtime_values[sample_index])
            result[view][method] = method_metrics

    return result


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reconstruct_comparison_samples(
    project_root: str | Path,
    *,
    view_levels: Sequence[int] = VIEW_LEVELS,
    seed: int = DEFAULT_SEED,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[int, np.ndarray],
    dict[int, dict[str, dict[str, float]]],
    str,
]:
    """Reproduce five seeded dataloader samples, one for each view count."""
    import deepinv as dinv
    import torch
    from deepinv.loss import LPIPS, PSNR, SSIM

    from src.optim.sampler_for_comparison import (
        DDSSampler,
        DiffPIRSampler,
        FBPSampler,
        Multi_Scale_DiffPIRSampler,
    )
    from src.physics.tomography import (
        DiagWeightPhysics,
        MultiScaleSinogram,
        MultiScaleTensor,
        MultiScaleTomography,
        Tomography,
        WeightedMultiScalePhysics,
    )
    from src.utils.dataloaders import get_att_ct_dataloaders
    from src.utils.load import compose_cfg, load_unet_diff, to_plain_dict

    project_root = Path(project_root).resolve()
    _set_seed(seed)
    cfg = compose_cfg(
        config_dir=project_root / "configs",
        config_name="config",
        overrides=[],
    )
    config = to_plain_dict(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=config["dataloader"]["loader"]["batch_size"],
        shuffle=True,
    )
    models = load_unet_diff(config["modelset"]["models"])
    model = models[0]
    diffusion_config = config["experiment"]["diffusion"]
    multiscale_sampler = Multi_Scale_DiffPIRSampler(
        **diffusion_config, device=device
    )
    single_sampler = DiffPIRSampler(**diffusion_config, device=device)
    dds_sampler = DDSSampler(**diffusion_config, device=device)
    fbp_sampler = FBPSampler()

    # Keep Comparison.py's construction order before consuming the dataloader.
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS(device=device, check_input_range=False)
    samples = [
        batch.to(device)[:1]
        for _, batch in zip(range(len(view_levels)), dataloader)
    ]
    sample_path = "seeded shuffled test dataloader samples 1-5"

    images: dict[int, dict[str, np.ndarray]] = {}
    ground_truths: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, dict[str, float]]] = {}
    intensity = 1e6
    data_fidelity = dinv.optim.L2()

    def quality_values(
        reconstruction_norm: torch.Tensor, x_true: torch.Tensor
    ) -> dict[str, float]:
        true_norm = x_true / ATTENUATION_MAX
        return {
            "psnr": float(psnr_metric(reconstruction_norm, true_norm).item()),
            "ssim": float(ssim_metric(reconstruction_norm, true_norm).item()),
            "lpips": float(
                lpips_metric(
                    reconstruction_norm.clamp(0.0, 1.0).repeat(1, 3, 1, 1),
                    true_norm.clamp(0.0, 1.0).repeat(1, 3, 1, 1),
                ).item()
            ),
        }

    def timed_call(function: Any, **kwargs: Any) -> tuple[Any, float]:
        quiet_output = StringIO()
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with redirect_stdout(quiet_output):
                result = function(**kwargs)
            end_event.record()
            torch.cuda.synchronize()
            return result, float(start_event.elapsed_time(end_event))
        started = time.perf_counter()
        with redirect_stdout(quiet_output):
            result = function(**kwargs)
        return result, (time.perf_counter() - started) * 1000.0

    with torch.no_grad():
        for row_index, view in enumerate(view_levels):
            print(f"Reconstructing seeded sample {row_index + 1} at {view} views...", flush=True)
            x_true = samples[row_index]
            noise_model = dinv.physics.LogPoissonNoise(
                N0=intensity, mu=1.0, rng=None
            )
            physics = Tomography(n_view=view, noise_model=noise_model)
            multiscale_physics = MultiScaleTomography(
                scales=(1, 2, 4),
                n_view=view,
                noise_model=noise_model,
                device=device,
            )
            measurement = physics(x_true).clamp_min(0)
            multiscale_measurement = MultiScaleSinogram(
                measurement, scales=(1, 2, 4)
            )

            (x_fbp, _), fbp_runtime = timed_call(
                fbp_sampler.sample,
                y=measurement,
                physics=physics,
                clamp_min=0.0,
                clamp_max=ATTENUATION_MAX,
            )
            x_fbp_norm = x_fbp / ATTENUATION_MAX

            weight_sqrt = (intensity * torch.exp(-measurement)).sqrt()
            weight_sqrt = weight_sqrt / weight_sqrt.mean()
            weight_physics = DiagWeightPhysics(wsqrt=weight_sqrt)
            weighted_physics = weight_physics * physics
            weighted_measurement = weight_physics.A(measurement)

            multiscale_levels = [
                multiscale_measurement.get_measurement(level)
                for level in range(multiscale_physics.n_levels)
            ]
            weight_sqrt_levels = []
            weighted_measurement_levels = []
            for measurement_level in multiscale_levels:
                level_weight = (intensity * torch.exp(-measurement_level)).sqrt()
                level_weight = level_weight / level_weight.mean()
                weight_sqrt_levels.append(level_weight)
                weighted_measurement_levels.append(level_weight * measurement_level)
            weighted_multiscale_measurement = MultiScaleTensor(
                weighted_measurement_levels
            )
            weighted_multiscale_physics = WeightedMultiScalePhysics(
                base_physics=multiscale_physics,
                wsqrt_levels=weight_sqrt_levels,
            )

            (x_multiscale_norm, _), multiscale_runtime = timed_call(
                multiscale_sampler.sample,
                y=weighted_multiscale_measurement,
                physics=weighted_multiscale_physics,
                data_fidelity=data_fidelity,
                lambda_=1e-3,
                model=models,
                levels=[2, 1, 0],
                transition_ts=[70, 30],
                x_init=x_fbp_norm,
                t_start=199,
                noise=None,
                num_ddim_steps=50,
                zeta=0.4,
                record_history=False,
            )

            (x_single_norm, _), single_runtime = timed_call(
                single_sampler.sample,
                y=weighted_measurement,
                physics=weighted_physics,
                data_fidelity=data_fidelity,
                lambda_=1e-3,
                model=model,
                x_init=x_fbp_norm,
                t_start=199,
                noise=None,
                num_ddim_steps=50,
                zeta=0.4,
                record_history=False,
            )

            (x_dds_norm, _), dds_runtime = timed_call(
                dds_sampler.sample,
                y=weighted_measurement,
                physics=weighted_physics,
                data_fidelity=data_fidelity,
                lambda_=1e-3,
                model=model,
                x_init=x_fbp_norm,
                t_start=199,
                noise=None,
                num_ddim_steps=50,
                zeta=0.4,
                record_history=False,
                prox_gamma=1e16,
                prox_max_iter=10,
            )

            normalized = {
                "FBP": x_fbp_norm,
                "DDS": x_dds_norm,
                "Single-scale DiffPIR": x_single_norm,
                "Multiscale DiffPIR": x_multiscale_norm,
            }
            ground_truths[view] = (
                x_true[0, 0].detach().cpu().numpy().astype(np.float32)
            )
            images[view] = {
                method: (ATTENUATION_MAX * value.clamp(0.0, 1.0))[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                for method, value in normalized.items()
            }
            metrics[view] = {}
            runtime_values = {
                "FBP": fbp_runtime,
                "DDS": dds_runtime,
                "Single-scale DiffPIR": single_runtime,
                "Multiscale DiffPIR": multiscale_runtime,
            }
            for method, value in normalized.items():
                metrics[view][method] = quality_values(value, x_true)
                metrics[view][method]["runtime_ms"] = runtime_values[method]

    return images, ground_truths, metrics, sample_path


def save_reconstruction_data(
    data_path: str | Path,
    images: Mapping[int, Mapping[str, np.ndarray]],
    ground_truths: Mapping[int, np.ndarray],
    metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    seed: int,
    sample_description: str,
) -> None:
    """Save all reconstruction arrays and metrics needed for later replotting."""
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "seed": np.asarray(seed, dtype=np.int64),
        "view_levels": np.asarray(VIEW_LEVELS, dtype=np.int64),
        "sample_description": np.asarray(sample_description),
    }
    for view in VIEW_LEVELS:
        archive[f"gt_{view}"] = np.asarray(ground_truths[view], dtype=np.float32)
        for method in METHODS[:-1]:
            key = _METHOD_TO_ARCHIVE_KEY[method]
            archive[f"image_{view}_{key}"] = np.asarray(
                images[view][method], dtype=np.float32
            )
            method_metrics = metrics[view][method]
            archive[f"metrics_{view}_{key}"] = np.asarray(
                [
                    method_metrics["psnr"],
                    method_metrics["ssim"],
                    method_metrics["lpips"],
                    method_metrics["runtime_ms"],
                ],
                dtype=np.float64,
            )
    np.savez_compressed(data_path, **archive)


def load_reconstruction_data(
    data_path: str | Path,
    *,
    expected_seed: int,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[int, np.ndarray],
    dict[int, dict[str, dict[str, float]]],
    str,
]:
    """Load cached reconstruction arrays, rejecting incompatible cache files."""
    data_path = Path(data_path)
    with np.load(data_path, allow_pickle=False) as archive:
        cached_seed = int(archive["seed"])
        cached_views = tuple(int(view) for view in archive["view_levels"])
        if cached_seed != expected_seed:
            raise ValueError(
                f"Cache seed {cached_seed} does not match requested seed {expected_seed}."
            )
        if cached_views != VIEW_LEVELS:
            raise ValueError(
                f"Cache views {cached_views} do not match required views {VIEW_LEVELS}."
            )

        images: dict[int, dict[str, np.ndarray]] = {}
        ground_truths: dict[int, np.ndarray] = {}
        metrics: dict[int, dict[str, dict[str, float]]] = {}
        for view in VIEW_LEVELS:
            ground_truths[view] = archive[f"gt_{view}"].copy()
            images[view] = {}
            metrics[view] = {}
            for method in METHODS[:-1]:
                key = _METHOD_TO_ARCHIVE_KEY[method]
                images[view][method] = archive[f"image_{view}_{key}"].copy()
                values = np.asarray(archive[f"metrics_{view}_{key}"])
                metrics[view][method] = {
                    "psnr": float(values[0]),
                    "ssim": float(values[1]),
                    "lpips": float(values[2]),
                    "runtime_ms": float(values[3]),
                }
        sample_description = str(archive["sample_description"])
    return images, ground_truths, metrics, sample_description


def _format_metrics(metrics: Mapping[str, Any] | Sequence[float]) -> str:
    if isinstance(metrics, Mapping):
        psnr = float(metrics["psnr"])
        ssim = float(metrics["ssim"])
        lpips = float(metrics["lpips"])
        runtime_ms = float(metrics["runtime_ms"])
    else:
        if len(metrics) != 4:
            raise ValueError("Metric sequences must contain PSNR, SSIM, LPIPS, runtime_ms.")
        psnr, ssim, lpips, runtime_ms = map(float, metrics)
    return f"{psnr:.1f} / {ssim:.3f} / {lpips:.3f} / {runtime_ms / 1000.0:.1f}s"


def load_quantitative_averages(
    outputs_dir: str | Path,
    *,
    view_levels: Sequence[int] = VIEW_LEVELS,
) -> dict[int, dict[str, dict[str, float]]]:
    """Load exact 500-image final-metric and timing averages from comparison runs."""
    outputs_dir = Path(outputs_dir)
    metric_keys = {
        "FBP": "fbp",
        "DDS": "dds",
        "Single-scale DiffPIR": "single_scale",
        "Multiscale DiffPIR": "multi_scale",
    }
    averages: dict[int, dict[str, dict[str, float]]] = {}

    for view in view_levels:
        eval_dir = outputs_dir / "comparison_runs" / f"nview_{view}_evals_500"
        time_path = (
            outputs_dir / "comparison_runs" / f"nview_{view}_time_500"
            / "time" / "time_per_image.npz"
        )
        metric_archives: dict[str, dict[str, np.ndarray]] = {}
        for metric in ("psnr", "ssim", "lpips"):
            metric_path = eval_dir / metric / f"{metric}_raw_curves.npz"
            if not metric_path.is_file():
                raise FileNotFoundError(f"Missing experimental data: {metric_path}")
            with np.load(metric_path, allow_pickle=True) as archive:
                metric_archives[metric] = {
                    key: archive[key].copy() for key in archive.files
                }
        if not time_path.is_file():
            raise FileNotFoundError(f"Missing experimental data: {time_path}")

        averages[view] = {}
        with np.load(time_path, allow_pickle=False) as timing:
            for method, archive_key in metric_keys.items():
                method_values: dict[str, float] = {}
                for metric, archive in metric_archives.items():
                    values = archive[archive_key]
                    if len(values) != 500:
                        raise ValueError(
                            f"Expected 500 {metric} values for {method}, {view} views; "
                            f"found {len(values)}."
                        )
                    if archive_key == "fbp":
                        final_values = np.asarray(values, dtype=np.float64)
                    else:
                        final_values = np.asarray(
                            [float(track[min(track)]) for track in values],
                            dtype=np.float64,
                        )
                    method_values[metric] = float(final_values.mean())

                runtime_values = np.asarray(
                    timing[_METHOD_TO_TIME_KEY[method]], dtype=np.float64
                )
                if len(runtime_values) != 500:
                    raise ValueError(
                        f"Expected 500 runtime values for {method}, {view} views; "
                        f"found {len(runtime_values)}."
                    )
                method_values["time"] = float(runtime_values.mean() / 1000.0)
                averages[view][method] = method_values
    return averages


def plot_quantitative_comparison_table(
    averages: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    output_path: str | Path,
    view_levels: Sequence[int] = VIEW_LEVELS,
    methods: Sequence[str] = METHODS[:-1],
    dpi: int = 300,
) -> plt.Figure:
    """Render a booktabs-style quantitative comparison table."""
    metric_specs = (
        ("psnr", "PSNR ↑", ".2f", True),
        ("ssim", "SSIM ↑", ".4f", True),
        ("lpips", "LPIPS ↓", ".4f", False),
        ("time", "Time ↓", ".2f", False),
    )
    method_width = 2.65
    metric_width = 1.18
    total_width = method_width + len(view_levels) * len(metric_specs) * metric_width
    total_height = 2 + len(methods)
    figure, axis = plt.subplots(figsize=(24, 4.25))
    axis.set_xlim(0, total_width)
    axis.set_ylim(0, total_height)
    axis.axis("off")

    axis.plot((0, total_width), (total_height, total_height), color="black", lw=1.6)
    axis.plot((0, total_width), (len(methods), len(methods)), color="black", lw=0.9)
    axis.plot((0, total_width), (0, 0), color="black", lw=1.6)
    axis.text(
        method_width / 2, len(methods) + 1, "Method",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
        fontfamily="serif",
    )

    for view_index, view in enumerate(view_levels):
        group_left = method_width + view_index * len(metric_specs) * metric_width
        group_right = group_left + len(metric_specs) * metric_width
        axis.text(
            (group_left + group_right) / 2, len(methods) + 1.5,
            f"{view} views", ha="center", va="center", fontsize=11.5,
            fontweight="bold", fontfamily="serif",
        )
        axis.plot(
            (group_left + 0.10, group_right - 0.10),
            (len(methods) + 1.02, len(methods) + 1.02),
            color="black", lw=0.7,
        )
        for metric_index, (_, label, _, _) in enumerate(metric_specs):
            axis.text(
                group_left + (metric_index + 0.5) * metric_width,
                len(methods) + 0.5, label, ha="center", va="center",
                fontsize=9.5, fontfamily="serif",
            )

    display_names = {
        "Single-scale DiffPIR": "DiffPIR",
        "Multiscale DiffPIR": "Multiscale DiffPIR",
    }
    for row_index, method in enumerate(methods):
        axis.text(
            0.14, len(methods) - row_index - 0.5,
            display_names.get(method, method), ha="left", va="center",
            fontsize=10.5, fontfamily="serif",
        )

    for view_index, view in enumerate(view_levels):
        group_left = method_width + view_index * len(metric_specs) * metric_width
        for metric_index, (metric, _, number_format, higher_is_better) in enumerate(metric_specs):
            raw_values = np.asarray(
                [averages[view][method][metric] for method in methods],
                dtype=np.float64,
            )
            ranking = np.argsort(-raw_values if higher_is_better else raw_values)
            x_center = group_left + (metric_index + 0.5) * metric_width
            for row_index, value in enumerate(raw_values):
                y_center = len(methods) - row_index - 0.5
                text = axis.text(
                    x_center, y_center, format(float(value), number_format),
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if row_index == int(ranking[0]) else "normal",
                    fontfamily="serif",
                )
                if row_index == int(ranking[1]):
                    figure.canvas.draw()
                    bounds = text.get_window_extent(renderer=figure.canvas.get_renderer())
                    left, bottom = axis.transData.inverted().transform((bounds.x0, bounds.y0))
                    right, _ = axis.transData.inverted().transform((bounds.x1, bounds.y0))
                    axis.plot(
                        (left, right), (bottom - 0.025, bottom - 0.025),
                        color="black", lw=0.65,
                    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    return figure


def regenerate_average_metric_plots(outputs_dir: str | Path, *, dpi: int = 300) -> int:
    """Regenerate every saved average metric plot with the label ``DiffPIR``."""
    outputs_dir = Path(outputs_dir)
    metric_settings = {
        "psnr": ("PSNR", "Average PSNR (dB)"),
        "ssim": ("SSIM", "Average SSIM"),
        "lpips": ("LPIPS", "Average LPIPS"),
    }
    raw_paths = sorted(outputs_dir.glob("comparison_runs/nview_*_evals_500/*/*_raw_curves.npz"))
    raw_paths.extend(
        outputs_dir / metric / f"{metric}_raw_curves.npz"
        for metric in metric_settings
        if (outputs_dir / metric / f"{metric}_raw_curves.npz").is_file()
    )
    generated = 0
    for raw_path in raw_paths:
        metric = raw_path.parent.name.lower()
        if metric not in metric_settings:
            continue
        metric_name, average_ylabel = metric_settings[metric]
        with np.load(raw_path, allow_pickle=True) as archive:
            single_tracks = archive["single_scale"]
            multi_tracks = archive["multi_scale"]
            dds_tracks = archive["dds"]
            fbp_values = np.asarray(archive["fbp"], dtype=np.float64)
        sample_count = len(single_tracks)
        if not (
            len(multi_tracks) == len(dds_tracks) == len(fbp_values) == sample_count
        ):
            raise ValueError(f"Inconsistent sample counts in {raw_path}.")

        figure, axis = plt.subplots(figsize=(8, 5))
        for tracks, label, color, linestyle in (
            (single_tracks, "Average DiffPIR (Single-scale)", "blue", "-"),
            (multi_tracks, "Average Proposed (Multi-scale, 512-stage)", "red", "--"),
            (dds_tracks, "Average DDS", "purple", "-."),
        ):
            timesteps = sorted(tracks[0].keys(), reverse=True)
            values = [np.mean([track[timestep] for track in tracks]) for timestep in timesteps]
            axis.plot(
                timesteps, values, label=label, color=color,
                linestyle=linestyle, linewidth=2.5,
            )
        axis.axhline(
            float(fbp_values.mean()), label="Average FBP", color="green",
            linestyle=":", linewidth=2.5,
        )
        axis.invert_xaxis()
        axis.set_xlabel("Timestep $t$")
        axis.set_ylabel(average_ylabel)
        axis.set_title(
            f"Statistical Average {metric_name} Convergence Curve "
            f"({sample_count} Images)"
        )
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.6)

        if "comparison_runs" in raw_path.parts:
            output_path = raw_path.parents[1] / "figures" / metric / f"{metric}_curve_average_total.png"
        else:
            output_path = raw_path.parent / f"{metric}_curve_average_total.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        generated += 1
    return generated


def plot_reconstruction_matrix(
    images_dict: Mapping[int, Mapping[str, np.ndarray]],
    gt_images: Mapping[int, np.ndarray],
    roi_coords: Mapping[int, Sequence[int]],
    metrics_dict: Mapping[int, Mapping[str, Mapping[str, float] | Sequence[float]]],
    *,
    view_levels: Sequence[int] = VIEW_LEVELS,
    methods: Sequence[str] = METHODS,
    image_vmin: float = 0.0,
    image_vmax: float = 0.04825,
    error_vmin: float = 0.0,
    error_vmax: float = 0.01,
    error_cmap: str = "magma",
    roi_display_size: int = 256,
    output_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot full reconstructions, enlarged ROIs, and absolute ROI errors.

    The rows are view counts and columns are reconstruction methods plus GT.
    Each metric heading is a single line. Error panels use the same color scale,
    shown beside each row's zero-valued GT error panel.

    Arrays may have shape ``(H, W)``, ``(1, H, W)``, or ``(1, 1, H, W)``.
    ``image_vmax=0.04825`` matches this project's attenuation-value convention;
    use ``image_vmax=1.0`` and the desired ``error_vmax`` for normalized arrays.
    """
    if methods[-1] != "GT":
        raise ValueError("The last method must be 'GT'.")
    if image_vmax <= image_vmin or error_vmax <= error_vmin:
        raise ValueError("Display maxima must be greater than display minima.")
    if roi_display_size <= 0:
        raise ValueError("roi_display_size must be positive.")

    row_count = len(view_levels)
    column_count = len(methods)
    figure = plt.figure(
        figsize=(2.75 * column_count + 0.45, 3.72 * row_count),
        constrained_layout=False,
    )
    outer_grid = figure.add_gridspec(
        row_count,
        column_count,
        left=0.055,
        right=0.955,
        bottom=0.055,
        top=0.98,
        wspace=0.006,
        hspace=0.02,
    )
    axes = np.empty((row_count, column_count, 3), dtype=object)

    for row_index, view in enumerate(view_levels):
        if view not in images_dict or view not in gt_images or view not in roi_coords:
            raise KeyError(f"Missing images, GT, or ROI for {view} views.")

        gt = _as_2d_float(gt_images[view], f"GT for {view} views")
        x, y, width, height = _validate_roi(roi_coords[view], gt.shape, view)
        gt_roi = gt[y : y + height, x : x + width]
        gt_roi_display = _resize_roi(gt_roi, roi_display_size)

        for column_index, method in enumerate(methods):
            inner_grid = outer_grid[row_index, column_index].subgridspec(
                3, 2, height_ratios=(0.24, 2.05, 1.0), wspace=0.015, hspace=0.0
            )
            metric_axis = figure.add_subplot(inner_grid[0, :])
            full_axis = figure.add_subplot(inner_grid[1, :])
            roi_axis = figure.add_subplot(inner_grid[2, 0])
            error_axis = figure.add_subplot(inner_grid[2, 1])
            axes[row_index, column_index] = (full_axis, roi_axis, error_axis)

            if method == "GT":
                image = gt
            else:
                if method not in images_dict[view]:
                    raise KeyError(f"Missing {method} reconstruction for {view} views.")
                image = _as_2d_float(
                    images_dict[view][method], f"{method} for {view} views"
                )
                if image.shape != gt.shape:
                    raise ValueError(
                        f"{method}, {view} views has shape {image.shape}; "
                        f"expected {gt.shape}."
                    )

            full_axis.imshow(
                image, cmap="gray", vmin=image_vmin, vmax=image_vmax,
                interpolation="nearest"
            )
            image_roi = image[y : y + height, x : x + width]
            roi_display = _resize_roi(image_roi, roi_display_size)
            error_display = cv2.absdiff(roi_display, gt_roi_display)

            roi_axis.imshow(
                roi_display, cmap="gray", vmin=image_vmin, vmax=image_vmax,
                interpolation="nearest"
            )
            error_axis.imshow(
                error_display,
                cmap=error_cmap,
                vmin=error_vmin,
                vmax=error_vmax,
                interpolation="nearest",
            )

            if method == "GT":
                full_axis.add_patch(
                    Rectangle(
                        (x, y), width, height, fill=False,
                        edgecolor="#e31a1c", linewidth=1.4
                    )
                )
                metric_text = "PSNR / SSIM / LPIPS / RUNTIME"
            else:
                try:
                    metrics = metrics_dict[view][method]
                except KeyError as error:
                    raise KeyError(
                        f"Missing metrics for {method}, {view} views."
                    ) from error
                metric_text = _format_metrics(metrics)

            metric_axis.set_facecolor("black")
            metric_axis.text(
                0.5,
                0.5,
                metric_text,
                transform=metric_axis.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=7.6,
                fontweight="bold",
            )

            if method == "GT":
                colorbar_axis = error_axis.inset_axes((0.72, 0.20, 0.11, 0.60))
                colorbar = figure.colorbar(
                    ScalarMappable(
                        norm=Normalize(vmin=error_vmin, vmax=error_vmax),
                        cmap=error_cmap,
                    ),
                    cax=colorbar_axis,
                    ticks=(error_vmin, error_vmax),
                )
                colorbar.ax.set_yticklabels(
                    (f"{error_vmin:.2f}", f"{error_vmax:g}")
                )
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
                colorbar.ax.tick_params(
                    axis="y",
                    colors="white",
                    labelsize=6.5,
                    length=2,
                    width=0.8,
                    pad=1.5,
                )
                colorbar.outline.set_edgecolor("white")
                colorbar.outline.set_linewidth(0.8)

            if column_index == 0:
                full_axis.text(
                    -0.10,
                    0.5,
                    f"{view} views",
                    transform=full_axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                )

            for axis in (full_axis, roi_axis, error_axis):
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            metric_axis.set_xticks([])
            metric_axis.set_yticks([])
            for spine in metric_axis.spines.values():
                spine.set_visible(False)

            # imshow keeps the full CT image square inside its grid cell. Match
            # the metric strip to that final image width and attach it directly.
            image_position = full_axis.get_position()
            metric_position = metric_axis.get_position()
            column_gap = 0.003
            compact_width = (
                image_position.width * column_count
                + column_gap * (column_count - 1)
            )
            compact_left = 0.5 - compact_width / 2.0
            compact_x = compact_left + column_index * (
                image_position.width + column_gap
            )
            full_axis.set_position(
                (
                    compact_x,
                    image_position.y0,
                    image_position.width,
                    image_position.height,
                )
            )
            image_position = full_axis.get_position()
            metric_axis.set_position(
                (
                    image_position.x0,
                    image_position.y1,
                    image_position.width,
                    metric_position.height,
                )
            )

            # Make the two square detail panels exactly fill the main image
            # width. Their shared edge is identical, so no white seam remains.
            detail_position = roi_axis.get_position()
            detail_size = image_position.width / 2.0
            roi_axis.set_position(
                (
                    image_position.x0,
                    detail_position.y0,
                    detail_size,
                    detail_size,
                )
            )
            error_axis.set_position(
                (
                    image_position.x0 + detail_size,
                    detail_position.y0,
                    detail_size,
                    detail_size,
                )
            )

            if row_index == row_count - 1:
                roi_axis.text(
                    1.0,
                    -0.13,
                    _METHOD_DISPLAY_NAMES.get(method, method),
                    transform=roi_axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    clip_on=False,
                )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show()

    return figure, axes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the seeded first CT sample and create the paper figure."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="ROI in image coordinates; defaults to an automatic high-detail ROI.",
    )
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--error-max", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction_comparison.png"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "Cached NPZ data path; defaults to '<output stem>_data.npz' in the "
            "same directory as the PNG."
        ),
    )
    parser.add_argument(
        "--force-reconstruct",
        action="store_true",
        help="Ignore any cached NPZ file and rerun all reconstructions.",
    )
    parser.add_argument(
        "--quantitative-table",
        action="store_true",
        help="Render the 500-image quantitative table from comparison_runs.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("outputs/quantitative_comparison_table.png"),
    )
    parser.add_argument(
        "--regenerate-average-plots",
        action="store_true",
        help="Regenerate all average metric plots using the label DiffPIR.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.regenerate_average_plots:
        generated = regenerate_average_metric_plots(project_root / "outputs")
        print(f"Regenerated {generated} average plots with the label DiffPIR.")
        return

    if args.quantitative_table:
        averages = load_quantitative_averages(project_root / "outputs")
        figure = plot_quantitative_comparison_table(
            averages, output_path=args.table_output
        )
        plt.close(figure)
        print("Averaged 500 experimental values per method/view/metric.")
        print(f"Saved quantitative table: {args.table_output.resolve()}")
        return

    data_path = args.data or args.output.with_name(f"{args.output.stem}_data.npz")
    if data_path.is_file() and not args.force_reconstruct:
        try:
            images, ground_truths, metrics, sample_path = load_reconstruction_data(
                data_path, expected_seed=args.seed
            )
            print(f"Loaded cached reconstruction data: {data_path.resolve()}")
        except (KeyError, ValueError) as error:
            print(f"Ignoring incompatible cache ({error}); reconstructing...")
            images, ground_truths, metrics, sample_path = reconstruct_comparison_samples(
                project_root, seed=args.seed
            )
            save_reconstruction_data(
                data_path,
                images,
                ground_truths,
                metrics,
                seed=args.seed,
                sample_description=sample_path,
            )
    else:
        images, ground_truths, metrics, sample_path = reconstruct_comparison_samples(
            project_root, seed=args.seed
        )
        save_reconstruction_data(
            data_path,
            images,
            ground_truths,
            metrics,
            seed=args.seed,
            sample_description=sample_path,
        )
        print(f"Saved reconstruction data: {data_path.resolve()}")
    if args.roi:
        rois = {view: tuple(args.roi) for view in VIEW_LEVELS}
    else:
        rois: dict[int, tuple[int, int, int, int]] = {}
        selected_rois: list[tuple[int, int, int, int]] = []
        fixed_rois = {
            20: (205, 165, args.roi_size, args.roi_size),
            40: (300, 300, args.roi_size, args.roi_size),
            60: (245, 245, args.roi_size, args.roi_size),
            100: (220, 250, args.roi_size, args.roi_size),
        }
        for view in VIEW_LEVELS:
            roi = fixed_rois.get(view)
            if roi is None:
                roi = select_detail_roi(
                    ground_truths[view],
                    roi_size=args.roi_size,
                    excluded_rois=selected_rois,
                )
            rois[view] = roi
            selected_rois.append(roi)
    print(f"Sample: {sample_path}")
    print(f"ROIs: {rois}")

    figure, _ = plot_reconstruction_matrix(
        images,
        ground_truths,
        rois,
        metrics,
        error_vmax=args.error_max,
        output_path=args.output,
    )
    plt.close(figure)
    print(f"Saved PNG: {args.output.resolve()}")


if __name__ == "__main__":
    main()