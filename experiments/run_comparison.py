import random
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import numpy as np
import torch

from src.physics.tomography import Tomography, MultiScaleTomography, DiagWeightPhysics, MultiScaleSinogram, MultiScaleTensor, WeightedMultiScalePhysics

# For comparison with the original DiffPIR implementation
from src.optim.sampler_for_comparison import DDSSampler, DiffPIRSampler, FBPSampler, Multi_Scale_DiffPIRSampler

from src.utils.load import load_unet_diff
from src.utils.dataloaders import get_att_ct_dataloaders
from src.utils.load import compose_cfg, to_plain_dict
from deepinv.loss import PSNR, SSIM, LPIPS
import matplotlib.pyplot as plt


def elapsed_time_ms(start_event, end_event, start_cpu_time, end_cpu_time, use_cuda):
    if use_cuda:
        return start_event.elapsed_time(end_event)
    return (end_cpu_time - start_cpu_time) * 1000.0

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_metric_curves(
    all_images_single_scale,
    all_images_multi_scale,
    all_images_dds,
    all_images_fbp,
    metric_name,
    ylabel,
    average_ylabel,
    output_dir,
):
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    metric_name_lower = metric_name.lower()
    num_saved_images = len(all_images_single_scale)

    for idx in range(num_saved_images):
        plt.figure(figsize=(8, 5))
        
        # extract dict value for the current image 
        # (notice all_images_single_scale and all_images_multi_scale are lists of dicts, 
        # so single_track and multi_track are dicts that we can directly plot)
        single_track = all_images_single_scale[idx]
        multi_track = all_images_multi_scale[idx]
        dds_track = all_images_dds[idx]
        fbp_value = all_images_fbp[idx]
        
        # plot single-scale full-stage curve, from t_start to 0
        plt.plot(list(single_track.keys()), list(single_track.values()), 
                    label='Single-scale DiffPIR', color='blue', linewidth=1.5)
        
        # plot multi-scale last 512 resolution stage curve, from final_stage_start_t to 0
        plt.plot(list(multi_track.keys()), list(multi_track.values()), 
                    label='Multi-scale DiffPIR, 512-stage', color='red', linestyle='--', linewidth=1.5)

        plt.plot(list(dds_track.keys()), list(dds_track.values()),
                label='DDS', color='purple', linestyle='-.', linewidth=1.5)

        plt.axhline(fbp_value,
                label='FBP', color='green', linestyle=':', linewidth=1.5)
        
        # reverse x-axis
        plt.gca().invert_xaxis() 
        
        plt.xlabel("Timestep $t$")
        plt.ylabel(ylabel)
        plt.title(f"{metric_name} Convergence Curve - Image {idx + 1}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.savefig(f"{output_dir}/{metric_name_lower}_curve_image_{idx + 1}.png", dpi=300, bbox_inches='tight')
        plt.close()


    plt.figure(figsize=(8, 5))

    # 1. Calculate the average metric at each time step across all images for single scale
    all_ts_single = sorted(all_images_single_scale[0].keys(), reverse=True) 
    avg_single = []
    for t in all_ts_single:
        # extract the metric values at time step t for all images and compute their average
        t_avg = np.mean([track[t] for track in all_images_single_scale])
        avg_single.append(t_avg)
        
    plt.plot(all_ts_single, avg_single, 
            label='Average Original DiffPIR (Single-scale)', color='blue', linewidth=2.5)


    # 2. Calculate and plot the average curve for multi-scale
    all_ts_multi = sorted(all_images_multi_scale[0].keys(), reverse=True)
    avg_multi = []
    for t in all_ts_multi:
        t_avg = np.mean([track[t] for track in all_images_multi_scale])
        avg_multi.append(t_avg)
        
    plt.plot(all_ts_multi, avg_multi, 
            label='Average Proposed (Multi-scale, 512-stage)', color='red', linestyle='--', linewidth=2.5)

    all_ts_dds = sorted(all_images_dds[0].keys(), reverse=True)
    avg_dds = []
    for t in all_ts_dds:
        t_avg = np.mean([track[t] for track in all_images_dds])
        avg_dds.append(t_avg)

    plt.plot(all_ts_dds, avg_dds,
            label='Average DDS', color='purple', linestyle='-.', linewidth=2.5)

    avg_fbp = np.mean(all_images_fbp)
    plt.axhline(avg_fbp,
                label='Average FBP', color='green', linestyle=':', linewidth=2.5)

    # reverse x-axis
    plt.gca().invert_xaxis() 
    plt.xlabel("Timestep $t$")
    plt.ylabel(average_ylabel)
    plt.title(f"Statistical Average {metric_name} Convergence Curve ({num_saved_images} Images)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(f"{output_dir}/{metric_name_lower}_curve_average_total.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{num_saved_images} {metric_name} plots and 1 average {metric_name} plot generated.")


def serialize_metric_tracks(metric_tracks):
    return np.array(
        [
            {
                int(t): float(v)
                for t, v in track.items()
            }
            for track in metric_tracks
        ],
        dtype=object,
    )


def save_metric_data_with_baselines(run_dir, metric_name, single_scale_tracks, multi_scale_tracks, dds_tracks, fbp_values):
    metric_name_lower = metric_name.lower()
    metric_dir = run_dir / metric_name_lower
    metric_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        metric_dir / f"{metric_name_lower}_raw_curves.npz",
        single_scale=serialize_metric_tracks(single_scale_tracks),
        multi_scale=serialize_metric_tracks(multi_scale_tracks),
        dds=serialize_metric_tracks(dds_tracks),
        fbp=np.array(fbp_values, dtype=np.float32),
    )


def save_run_config(run_dir, config_dict):
    config_path = run_dir / "run_config.pt"
    torch.save(config_dict, config_path)


def sync_file_to_latest(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    overrides = sys.argv[1:]

    cfg = compose_cfg(
        config_dir=project_root / "configs",
        config_name="config",
        overrides=overrides,
    )
    config = to_plain_dict(cfg)
    config_models = config['modelset']['models']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_timing = device.type == "cuda"
    set_seed(97)

    test_dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=config["dataloader"]["loader"]["batch_size"],
        shuffle=True,
    )

    models = load_unet_diff(config_models)
    # first one in the list: 512x512
    # second one in the list: 256x256
    # third one in the list: 128x128

    # let's generates in 512x512:
    model = models[0]

    multi_scale_sampler = Multi_Scale_DiffPIRSampler(**config["experiment"]["diffusion"], device=device)
    sampler = DiffPIRSampler(**config["experiment"]["diffusion"], device=device)
    dds_sampler = DDSSampler(**config["experiment"]["diffusion"], device=device)
    fbp_sampler = FBPSampler()
    
    I0 = 1e6 #just for some test #config["experiment"]["params"]["intensity"]
    noise_model = dinv.physics.LogPoissonNoise(N0=I0, mu=1.0, rng=None)
    data_fidelity = dinv.optim.L2() 
    physics = Tomography(n_view=20, noise_model=noise_model)


    multiscale_physics = MultiScaleTomography(scales=(1, 2, 4), n_view=20, noise_model=noise_model, device=device)


    psnr = PSNR()
    ssim = SSIM()
    lpips = LPIPS(device=device, check_input_range=False)
    all_images_psnr_single_scale = []   # All psnr dicts for single scale in a list
    all_images_psnr_multi_scale = []    # All psnr dicts for multiscale in a list
    all_images_ssim_single_scale = []
    all_images_ssim_multi_scale = []
    all_images_lpips_single_scale = []
    all_images_lpips_multi_scale = []
    all_images_psnr_dds = []
    all_images_ssim_dds = []
    all_images_lpips_dds = []
    all_images_psnr_fbp = []
    all_images_ssim_fbp = []
    all_images_lpips_fbp = []

    with torch.no_grad():
        max_images = 500
        # Firstly, set record_history = False and plot_time = True to record time
        # Secondly, set record_history = True and plot_time = False to record PSNR/SSIM/LPIPS curves
        # and save_raw_data = False is always True to save all the raw data for later analysis
        record_history = True  # Set False for pure timing without PSNR/SSIM/LPIPS curves.
        plot_time = False       # Set True to save timing data and timing scatter plot.
        save_raw_data = True  # Set True to save per-run raw data for later large-scale analysis.
        images_processed = 0

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"comparison_{run_timestamp}"
        run_dir = project_root / "outputs" / "comparison_runs" / run_name
        figures_dir = run_dir / "figures"
        latest_outputs_dir = project_root / "outputs"
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_run_config(
            run_dir,
            {
                "run_name": run_name,
                "timestamp": run_timestamp,
                "max_images": max_images,
                "record_history": record_history,
                "plot_time": plot_time,
                "save_raw_data": save_raw_data,
                "baselines": ["FBP", "DDS", "Original DiffPIR", "Multiscale DiffPIR"],
                "fbp_clamp_range": [0.0, 0.04825],
                "dds_prox_gamma": 1e16,
                "dds_prox_max_iter": 10,
                "levels": [2, 1, 0],
                "transition_ts": [70, 30],
                "t_start": 199,
                "num_ddim_steps": 50,
                "zeta": 0.4,
                "device": str(device),
                "use_cuda_timing": use_cuda_timing,
                "config": config,
                "overrides": overrides,
            },
        )

        total_time_ms_multiscale = 0.0
        total_time_ms_original = 0.0
        total_time_ms_dds = 0.0
        total_time_ms_fbp = 0.0

        time_ms_multiscale_per_image = []
        time_ms_original_per_image = []
        time_ms_dds_per_image = []
        time_ms_fbp_per_image = []
        image_indices = []

        levels = [2, 1, 0]
        transition_ts = [70, 30]
        final_stage_start_t = transition_ts[-1]
        t_start = 199

        for x in test_dataloader:

            x = x.to(device)

            # If the current batch would exceed the maximum number of images to process, 
            # only process the the number of images that we need in the batch
            remaining = max_images - images_processed
            if x.size(0) > remaining:
                x = x[:remaining]

            # noiseless line integrals
            # in deepinv, calling tomography like this when a noise model
            # is set give you a 'real measurement'
            b = physics(x).clamp_min(0)

            multiscale_b = MultiScaleSinogram(b, scales=(1, 2, 4))

            if use_cuda_timing:
                event_fbp_start = torch.cuda.Event(enable_timing=True)
                event_fbp_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                event_fbp_start.record()
            else:
                event_fbp_start = event_fbp_end = None
                time_start_fbp = time.perf_counter()

            x_fbp, _ = fbp_sampler.sample(b, physics, clamp_min=0.0, clamp_max=0.04825)

            if use_cuda_timing:
                event_fbp_end.record()
                torch.cuda.synchronize()
                time_fbp = elapsed_time_ms(event_fbp_start, event_fbp_end, None, None, use_cuda_timing)
            else:
                time_end_fbp = time.perf_counter()
                time_fbp = elapsed_time_ms(None, None, time_start_fbp, time_end_fbp, use_cuda_timing)

            # single scale b and wsqrt
            w = I0 * torch.exp(-b)
            wsqrt = w.sqrt()
            wsqrt = wsqrt / wsqrt.mean()
            Wsqrt = DiagWeightPhysics(wsqrt=wsqrt)

            weighted_physics = Wsqrt * physics  # For the original DiffPIR implementation
            
            b_tilde_single_scale = Wsqrt.A(b)

            # multiscale b and multiscale wsqrt for each level
            b_levels = [
                multiscale_b.get_measurement(level)
                for level in range(multiscale_physics.n_levels)
            ]

            wsqrt_levels = []
            b_tilde_levels = []

            for b_level in b_levels:
                w_level = I0 * torch.exp(-b_level)
                wsqrt_level = w_level.sqrt()
                wsqrt_level = wsqrt_level / wsqrt_level.mean()

                wsqrt_levels.append(wsqrt_level)
                b_tilde_levels.append(wsqrt_level * b_level)

            b_tilde_multiscale = MultiScaleTensor(b_tilde_levels)

            x_fbp_norm = x_fbp / 0.04825

            weighted_multiscale_physics = WeightedMultiScalePhysics(
                base_physics=multiscale_physics,
                wsqrt_levels=wsqrt_levels,
            )


            if use_cuda_timing:
                event1 = torch.cuda.Event(enable_timing=True)
                event2 = torch.cuda.Event(enable_timing=True)
                event3 = torch.cuda.Event(enable_timing=True)
                event4 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()    # Make sure all previous CUDA operations are finished before starting the timer
            else:
                event1 = event2 = event3 = event4 = None
            time_start_multiscale = None
            time_end_multiscale = None
            time_start_original = None
            time_end_original = None
            time_start_dds = None
            time_end_dds = None
            
            # Start multiscale timer
            if use_cuda_timing:
                event1.record()
            else:
                time_start_multiscale = time.perf_counter()

            _, history_multiscale = multi_scale_sampler.sample(y=b_tilde_multiscale,
                                physics=weighted_multiscale_physics,
                                data_fidelity=data_fidelity,
                                lambda_=1e-3,
                                model=models,
                                levels=levels,
                                transition_ts=transition_ts,
                                x_init=x_fbp_norm, t_start=t_start, noise=None,
                                num_ddim_steps = 50,
                                zeta = 0.4,
                                record_history=record_history,
                                )
            
            # End multiscale timer / Start original DiffPIR timer
            if use_cuda_timing:
                event2.record()
            else:
                time_end_multiscale = time.perf_counter()
                time_start_original = time_end_multiscale

            _, history_single_scale = sampler.sample(y=b_tilde_single_scale,
                                physics=weighted_physics,
                                data_fidelity=data_fidelity,
                                lambda_=1e-3,
                                model=model,
                                x_init=x_fbp_norm, t_start=t_start, noise=None,
                                num_ddim_steps = 50,
                                zeta = 0.4,
                                record_history=record_history,
                                )
            
            # End original DiffPIR timer / Start DDS timer
            if use_cuda_timing:
                event3.record()
            else:
                time_end_original = time.perf_counter()
                time_start_dds = time_end_original

            _, history_dds = dds_sampler.sample(y=b_tilde_single_scale,
                                physics=weighted_physics,
                                data_fidelity=data_fidelity,
                                lambda_=1e-3,
                                model=model,
                                x_init=x_fbp_norm, t_start=t_start, noise=None,
                                num_ddim_steps = 50,
                                zeta = 0.4,
                                record_history=record_history,
                                prox_gamma=1e16,
                                prox_max_iter=10,
                                )

            # End DDS timer
            if use_cuda_timing:
                event4.record()
            else:
                time_end_dds = time.perf_counter()

            if use_cuda_timing:
                torch.cuda.synchronize()

            time1 = elapsed_time_ms(event1, event2, time_start_multiscale, time_end_multiscale, use_cuda_timing)  # Time taken for multiscale sampler
            time2 = elapsed_time_ms(event2, event3, time_start_original, time_end_original, use_cuda_timing)  # Time taken for original DiffPIR sampler
            time_dds = elapsed_time_ms(event3, event4, time_start_dds, time_end_dds, use_cuda_timing)  # Time taken for DDS sampler
            
            # Update the counters
            total_time_ms_multiscale += time1
            total_time_ms_original += time2
            total_time_ms_dds += time_dds
            total_time_ms_fbp += time_fbp

            batch_size_current = x.size(0)
            time1_per_image = time1 / batch_size_current
            time2_per_image = time2 / batch_size_current
            time_dds_per_image = time_dds / batch_size_current
            time_fbp_per_image = time_fbp / batch_size_current

            for bi in range(batch_size_current):
                time_ms_multiscale_per_image.append(time1_per_image)
                time_ms_original_per_image.append(time2_per_image)
                time_ms_dds_per_image.append(time_dds_per_image)
                time_ms_fbp_per_image.append(time_fbp_per_image)
                image_indices.append(images_processed + bi + 1)

            images_processed += x.size(0)

            if record_history:
                x_true_metric = (x / 0.04825).detach().to(device)

                psnr_single_scale_per_image = [{} for _ in range(batch_size_current)]
                psnr_multi_scale_per_image = [{} for _ in range(batch_size_current)]
                ssim_single_scale_per_image = [{} for _ in range(batch_size_current)]
                ssim_multi_scale_per_image = [{} for _ in range(batch_size_current)]
                lpips_single_scale_per_image = [{} for _ in range(batch_size_current)]
                lpips_multi_scale_per_image = [{} for _ in range(batch_size_current)]
                psnr_dds_per_image = [{} for _ in range(batch_size_current)]
                ssim_dds_per_image = [{} for _ in range(batch_size_current)]
                lpips_dds_per_image = [{} for _ in range(batch_size_current)]
                psnr_fbp_per_image = []
                ssim_fbp_per_image = []
                lpips_fbp_per_image = []

                x_fbp_metric = x_fbp_norm.detach().to(device)

                for bi in range(batch_size_current):
                    x_fbp_bi = x_fbp_metric[bi:bi+1]
                    x_true_bi = x_true_metric[bi:bi+1]
                    psnr_fbp_per_image.append(psnr(x_fbp_bi, x_true_bi).item())
                    ssim_fbp_per_image.append(ssim(x_fbp_bi, x_true_bi).item())
                    lpips_fbp_per_image.append(lpips(x_fbp_bi.repeat(1, 3, 1, 1).clamp(0, 1), x_true_bi.repeat(1, 3, 1, 1).clamp(0, 1)).item())

                for t_step, x0_t in history_single_scale:
                    for bi in range(batch_size_current):
                        x0_t_bi = x0_t[bi:bi+1].to(device)
                        x_true_bi = x_true_metric[bi:bi+1]
                        psnr_single_scale_per_image[bi][t_step] = psnr(x0_t_bi, x_true_bi).item()
                        ssim_single_scale_per_image[bi][t_step] = ssim(x0_t_bi, x_true_bi).item()
                        lpips_single_scale_per_image[bi][t_step] = lpips(x0_t_bi.repeat(1, 3, 1, 1).clamp(0, 1), x_true_bi.repeat(1, 3, 1, 1).clamp(0, 1)).item()

                for t_step, x0_t in history_dds:
                    for bi in range(batch_size_current):
                        x0_t_bi = x0_t[bi:bi+1].to(device)
                        x_true_bi = x_true_metric[bi:bi+1]
                        psnr_dds_per_image[bi][t_step] = psnr(x0_t_bi, x_true_bi).item()
                        ssim_dds_per_image[bi][t_step] = ssim(x0_t_bi, x_true_bi).item()
                        lpips_dds_per_image[bi][t_step] = lpips(x0_t_bi.repeat(1, 3, 1, 1).clamp(0, 1), x_true_bi.repeat(1, 3, 1, 1).clamp(0, 1)).item()

                for t_step, x0_t in history_multiscale:
                    if t_step <= final_stage_start_t:
                        for bi in range(batch_size_current):
                            x0_t_bi = x0_t[bi:bi+1].to(device)
                            x_true_bi = x_true_metric[bi:bi+1]
                            psnr_multi_scale_per_image[bi][t_step] = psnr(x0_t_bi, x_true_bi).item()
                            ssim_multi_scale_per_image[bi][t_step] = ssim(x0_t_bi, x_true_bi).item()
                            lpips_multi_scale_per_image[bi][t_step] = lpips(x0_t_bi.repeat(1, 3, 1, 1).clamp(0, 1), x_true_bi.repeat(1, 3, 1, 1).clamp(0, 1)).item()

                all_images_psnr_single_scale.extend(psnr_single_scale_per_image)
                all_images_psnr_multi_scale.extend(psnr_multi_scale_per_image)
                all_images_ssim_single_scale.extend(ssim_single_scale_per_image)
                all_images_ssim_multi_scale.extend(ssim_multi_scale_per_image)
                all_images_lpips_single_scale.extend(lpips_single_scale_per_image)
                all_images_lpips_multi_scale.extend(lpips_multi_scale_per_image)
                all_images_psnr_dds.extend(psnr_dds_per_image)
                all_images_ssim_dds.extend(ssim_dds_per_image)
                all_images_lpips_dds.extend(lpips_dds_per_image)
                all_images_psnr_fbp.extend(psnr_fbp_per_image)
                all_images_ssim_fbp.extend(ssim_fbp_per_image)
                all_images_lpips_fbp.extend(lpips_fbp_per_image)

            print(f"Images processed: {images_processed} - FBP: {time_fbp:.2f} ms, DDS: {time_dds:.2f} ms, Multiscale DiffPIR: {time1:.2f} ms, Original DiffPIR: {time2:.2f} ms")
            
            if images_processed >= max_images:
                break
        
        average_time_ms_multiscale = np.mean(time_ms_multiscale_per_image)
        average_time_ms_original = np.mean(time_ms_original_per_image)
        average_time_ms_dds = np.mean(time_ms_dds_per_image)
        average_time_ms_fbp = np.mean(time_ms_fbp_per_image)

        print(f"Average time per image for FBP: {average_time_ms_fbp:.2f} ms")
        print(f"Average time per image for DDS: {average_time_ms_dds:.2f} ms")
        print(f"Average time per image for Multiscale DiffPIR Sampler: {average_time_ms_multiscale:.2f} ms")
        print(f"Average time per image for Original DiffPIR Sampler: {average_time_ms_original:.2f} ms")

        if save_raw_data:
            time_dir = run_dir / "time"
            time_dir.mkdir(parents=True, exist_ok=True)
            time_npz_path = time_dir / "time_per_image.npz"

            np.savez(
                time_npz_path,
                image_indices=np.array(image_indices),
                fbp_ms=np.array(time_ms_fbp_per_image),
                dds_ms=np.array(time_ms_dds_per_image),
                multiscale_ms=np.array(time_ms_multiscale_per_image),
                original_ms=np.array(time_ms_original_per_image),
                average_fbp_ms=average_time_ms_fbp,
                average_dds_ms=average_time_ms_dds,
                average_multiscale_ms=average_time_ms_multiscale,
                average_original_ms=average_time_ms_original,
                total_fbp_ms=total_time_ms_fbp,
                total_dds_ms=total_time_ms_dds,
                total_multiscale_ms=total_time_ms_multiscale,
                total_original_ms=total_time_ms_original,
            )

            sync_file_to_latest(time_npz_path, latest_outputs_dir / "time" / "time_per_image.npz")

        if plot_time:
            time_figure_dir = figures_dir / "time"
            time_figure_dir.mkdir(parents=True, exist_ok=True)
            time_figure_path = time_figure_dir / "time_per_image_scatter.png"

            plt.figure(figsize=(8, 5))

            plt.scatter(image_indices, time_ms_fbp_per_image,
                        label="FBP", color="green", s=18, alpha=0.7)

            plt.scatter(image_indices, time_ms_multiscale_per_image,
                        label="Multiscale DiffPIR", color="red", s=18, alpha=0.7)

            plt.scatter(image_indices, time_ms_dds_per_image,
                        label="DDS", color="purple", s=18, alpha=0.7)

            plt.scatter(image_indices, time_ms_original_per_image,
                        label="Original DiffPIR", color="blue", s=18, alpha=0.7)

            plt.axhline(average_time_ms_multiscale,
                        color="red", linestyle="--", linewidth=1.5,
                        label=f"Multiscale avg: {average_time_ms_multiscale:.2f} ms")

            plt.axhline(average_time_ms_original,
                        color="blue", linestyle="--", linewidth=1.5,
                        label=f"Original avg: {average_time_ms_original:.2f} ms")

            plt.axhline(average_time_ms_dds,
                        color="purple", linestyle="--", linewidth=1.5,
                        label=f"DDS avg: {average_time_ms_dds:.2f} ms")

            plt.axhline(average_time_ms_fbp,
                        color="green", linestyle="--", linewidth=1.5,
                        label=f"FBP avg: {average_time_ms_fbp:.2f} ms")

            plt.xlabel("Image index")
            plt.ylabel("Time per image (ms)")
            plt.title(f"Sampling Time per Image ({len(image_indices)} Images)")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)

            plt.savefig(time_figure_path, dpi=300, bbox_inches="tight")
            plt.close()

            sync_file_to_latest(time_figure_path, latest_outputs_dir / "time" / "time_per_image_scatter.png")

            print(f"Timing data and scatter plot generated for {len(image_indices)} images.")
    

        if record_history:
            if save_raw_data:
                save_metric_data_with_baselines(run_dir, "PSNR", all_images_psnr_single_scale, all_images_psnr_multi_scale, all_images_psnr_dds, all_images_psnr_fbp)
                save_metric_data_with_baselines(run_dir, "SSIM", all_images_ssim_single_scale, all_images_ssim_multi_scale, all_images_ssim_dds, all_images_ssim_fbp)
                save_metric_data_with_baselines(run_dir, "LPIPS", all_images_lpips_single_scale, all_images_lpips_multi_scale, all_images_lpips_dds, all_images_lpips_fbp)

                sync_file_to_latest(run_dir / "psnr" / "psnr_raw_curves.npz", latest_outputs_dir / "psnr" / "psnr_raw_curves.npz")
                sync_file_to_latest(run_dir / "ssim" / "ssim_raw_curves.npz", latest_outputs_dir / "ssim" / "ssim_raw_curves.npz")
                sync_file_to_latest(run_dir / "lpips" / "lpips_raw_curves.npz", latest_outputs_dir / "lpips" / "lpips_raw_curves.npz")

            plot_metric_curves(
                all_images_single_scale=all_images_psnr_single_scale,
                all_images_multi_scale=all_images_psnr_multi_scale,
                all_images_dds=all_images_psnr_dds,
                all_images_fbp=all_images_psnr_fbp,
                metric_name="PSNR",
                ylabel="PSNR (dB)",
                average_ylabel="Average PSNR (dB)",
                output_dir=figures_dir / "psnr",
            )

            for figure_path in (figures_dir / "psnr").glob("*.png"):
                sync_file_to_latest(figure_path, latest_outputs_dir / "psnr" / figure_path.name)

            plot_metric_curves(
                all_images_single_scale=all_images_ssim_single_scale,
                all_images_multi_scale=all_images_ssim_multi_scale,
                all_images_dds=all_images_ssim_dds,
                all_images_fbp=all_images_ssim_fbp,
                metric_name="SSIM",
                ylabel="SSIM",
                average_ylabel="Average SSIM",
                output_dir=figures_dir / "ssim",
            )

            for figure_path in (figures_dir / "ssim").glob("*.png"):
                sync_file_to_latest(figure_path, latest_outputs_dir / "ssim" / figure_path.name)

            plot_metric_curves(
                all_images_single_scale=all_images_lpips_single_scale,
                all_images_multi_scale=all_images_lpips_multi_scale,
                all_images_dds=all_images_lpips_dds,
                all_images_fbp=all_images_lpips_fbp,
                metric_name="LPIPS",
                ylabel="LPIPS",
                average_ylabel="Average LPIPS",
                output_dir=figures_dir / "lpips",
            )

            for figure_path in (figures_dir / "lpips").glob("*.png"):
                sync_file_to_latest(figure_path, latest_outputs_dir / "lpips" / figure_path.name)

        print(f"Run outputs saved to: {run_dir}")





if __name__=='__main__':
    main()
