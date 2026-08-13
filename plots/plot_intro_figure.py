"""Four-panel figure for the paper introduction / background section.

Panels:
    (a) Ground-truth clean CT slice  -- attenuation coefficients, obtained by
        HU conversion (mu = mu_water * (HU / 1000 + 1), mu_water = 0.0193)
        and clipping HU to [-1000, 1500], so values live in [0, 0.04825].
    (b) Full-view sinogram   (720 views) -- dense angular sampling.
    (c) Sparse-view sinogram (60 views)  -- angular undersampling.
    (d) FBP reconstruction from (c)      -- streak artifacts caused by the
        missing views, motivating learned/model-based reconstruction.

The first slice of the first test patient is used, with the same selection
logic as test_fbp.py / test_diffpir.py (seed 97, shuffle=False).

Layout:
    "pipeline" : 1x4 with arrows, GT -> full sinogram -> sparse sinogram -> FBP

Usage:
    python plot_intro_figure.py
"""

import random
import sys
from pathlib import Path

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
import torch

from src.physics.tomography import Tomography
from src.utils.dataloaders import get_att_ct_dataloaders
from src.utils.load import compose_cfg, to_plain_dict

# ---------------------------------------------------------------------------
# Figure options (edit as needed)
# ---------------------------------------------------------------------------
USE_NOISE = False             # True: LogPoisson photon noise (as in experiments)
                              # False: clean sinograms -> artifacts are purely
                              # due to angular undersampling (clearer for intro)
FULL_VIEWS = 720              # full angular sampling (Tomography default)
SPARSE_VIEWS = 60             # sparse view count (configs/experiment/sparse_view/60.yaml)

MU_WATER = 0.0193
HU_MIN, HU_MAX = -1000.0, 1500.0
ATTENUATION_MAX = MU_WATER * (HU_MAX / 1000.0 + 1.0)   # 0.04825

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "intro_figure"

PANEL_TITLES = [
    "Clean CT slice",
    "Full-view sinogram",
    "Sparse-view sinogram",
    "FBP reconstruction",
]


def set_seed(seed: int = 97) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Small helpers shared by both layouts
# ---------------------------------------------------------------------------
def _show_image(ax, image: np.ndarray, vmin: float, vmax: float) -> None:
    # aspect="auto" fills the axes so all four panels have identical size and
    # equal gaps; figsize is tuned so the axes stay close to square (no
    # visible distortion of the 512x512 slice)
    ax.imshow(
        image, cmap="gray", vmin=vmin, vmax=vmax,
        interpolation="nearest", aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _show_sinogram(ax, sino: np.ndarray, vmax: float) -> None:
    ax.imshow(
        sino, cmap="gray", aspect="auto", vmin=0.0, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_letter(ax, letter: str, fontsize: int = 14) -> None:
    ax.text(
        0.02, 0.98, f"({letter})", transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", va="top", ha="left",
        color="white",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="black")],
    )


def _panel_caption(fig, ax, text: str, fontsize: float = 10.5) -> None:
    bbox = ax.get_position()
    fig.text(
        (bbox.x0 + bbox.x1) / 2.0, bbox.y0 - 0.04, text,
        ha="center", va="top", fontsize=fontsize, color="#333333", style="italic",
    )


# ---------------------------------------------------------------------------
# Layout: 1x4 pipeline with arrows
# ---------------------------------------------------------------------------
def plot_pipeline(gt, b_full, b_sparse, x_fbp, out_path: Path) -> None:
    sino_vmax = float(b_full.max())

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.1))
    fig.subplots_adjust(wspace=0.01, bottom=0.22, top=0.94, left=0.02, right=0.98)

    panels = [
        (gt, "image"),
        (b_full, "sino"),
        (b_sparse, "sino"),
        (x_fbp, "image"),
    ]

    for ax, (data, kind) in zip(axes, panels):
        if kind == "image":
            _show_image(ax, data, vmin=0.0, vmax=ATTENUATION_MAX)
        else:
            _show_sinogram(ax, data, vmax=sino_vmax)

    # panel letters + captions
    for ax, letter, title in zip(axes, "abcd", PANEL_TITLES):
        _panel_letter(ax, letter)
        _panel_caption(fig, ax, title)

    # arrows between panels only (no text labels)
    fig.canvas.draw()
    for i in range(3):
        pos_l = axes[i].get_position()
        pos_r = axes[i + 1].get_position()
        y_mid = (pos_l.y0 + pos_l.y1) / 2.0
        axes[i].annotate(
            "", xy=(pos_r.x0 + 0.001, y_mid), xytext=(pos_l.x1 - 0.001, y_mid),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    overrides = sys.argv[1:]

    cfg = compose_cfg(
        config_dir=project_root / "configs",
        config_name="config",
        overrides=overrides,
    )
    config = to_plain_dict(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(97)

    test_dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=1,
        shuffle=False,
    )

    noise_model = dinv.physics.LogPoissonNoise(N0=1e6, mu=1.0, rng=None) if USE_NOISE else None

    full_physics = Tomography(n_view=FULL_VIEWS, noise_model=noise_model, device=device)
    sparse_physics = Tomography(n_view=SPARSE_VIEWS, noise_model=noise_model, device=device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x in test_dataloader:  # first batch = first slice of first test patient
            x = x.to(device)

            gt = x[0, 0].cpu().numpy()                    # (512, 512) attenuation coeffs

            b_full = full_physics(x).clamp_min(0)
            b_sparse = sparse_physics(x).clamp_min(0)
            x_fbp = sparse_physics.A_dagger(b_sparse).clamp(0, ATTENUATION_MAX)

            b_full_np = b_full[0, 0].cpu().numpy()        # (720, 1024)
            b_sparse_np = b_sparse[0, 0].cpu().numpy()    # (60, 1024)
            x_fbp_np = x_fbp[0, 0].cpu().numpy()          # (512, 512)

            print(f"GT slice shape:       {gt.shape}, range [{gt.min():.5f}, {gt.max():.5f}]")
            print(f"Full-view sinogram:   {b_full_np.shape}")
            print(f"Sparse-view sinogram: {b_sparse_np.shape}")
            print(f"FBP reconstruction:   {x_fbp_np.shape}")

            plot_pipeline(gt, b_full_np, b_sparse_np, x_fbp_np,
                          OUTPUT_DIR / "intro_pipeline.png")

            print(f"Saved figure to: {OUTPUT_DIR}")
            break  # one slice is enough for the intro figure


if __name__ == "__main__":
    main()
