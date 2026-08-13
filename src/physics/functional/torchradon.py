from __future__ import annotations

from typing import Union
import torch
import torch_radon as tr

def create_projector(
    src_dist: float = 595.0,      # Source-to-Isocenter Distance (SID)
    det_dist: float = 490.0,      # Isocenter-to-Detector Distance (IDD)
    det_spacing: float = 1.0,     # Physical spacing between detector elements
    det_count: int = 1024,        # Number of detector channels
    dx: float = 0.8,              # Voxel size in x-direction (mm)
    dy: float = 0.8,              # Voxel size in y-direction (mm)
    nx: int = 512,                # Reconstruction grid width (pixels)
    ny: int = 512,                # Reconstruction grid height (pixels)
    n_angles: int = 60,           # Number of projection views
    scale: float = 1.0,           # Global intensity scaling factor
    device: Union[str, torch.device] = "cuda",
) -> tuple[tr.FanBeam, float]:
    """
    Configure the fan-beam CT geometry and initialize the Radon operator.

    Args:
        src_dist: Source-to-isocenter distance.
        det_dist: Isocenter-to-detector distance.
        det_spacing: Physical spacing between detector elements.
        det_count: Number of detector channels.
        dx: Voxel size along x.
        dy: Voxel size along y.
        nx: Reconstruction width in pixels.
        ny: Reconstruction height in pixels.
        n_angles: Number of projection angles.
        scale: Global intensity scaling factor.
        device: Torch device or device string.

    Returns:
        A tuple `(radon, scale)` where `radon` is the configured fan-beam operator.
    """
    angles = torch.linspace(0, 2 * torch.pi, n_angles, device=device)

    volume = tr.Volume2D(voxel_size=(dy, dx))
    volume.set_size(height=ny, width=nx)

    radon = tr.FanBeam(
        det_count=det_count,
        src_dist=src_dist,
        det_dist=det_dist,
        det_spacing=det_spacing,
        angles=angles,
        volume=volume,
    )

    return radon, scale