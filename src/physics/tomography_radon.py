from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

import deepinv as dinv
from deepinv.physics import LinearPhysics
from typing_extensions import override

from src.physics.functional.torchradon import create_projector


class DiagWeightPhysics(dinv.physics.LinearPhysics):
    def __init__(self, wsqrt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("wsqrt", wsqrt)

    def A(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.wsqrt * z

    def A_adjoint(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.wsqrt * z

    @override
    def A_vjp(self, x: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A_adjoint(v, **kwargs)


class Tomography(LinearPhysics):
    """
    DeepInverse wrapper around a torch-radon fan-beam operator.
    """

    def __init__(
        self,
        *,
        src_dist: float = 595.0,
        det_dist: float = 490.0,
        det_spacing: float = 1.0,
        det_count: int = 1024,
        dx: float = 0.8,
        dy: float = 0.8,
        nx: int = 512,
        ny: int = 512,
        n_angles: int = 60,
        scale: float = 1.0,
        device: torch.device | str = "cuda",
        fbp_window: str = "hann",
        noise_model=None,
        **kwargs,
    ) -> None:
        super().__init__(noise_model=noise_model, **kwargs)

        radon, backend_scale = create_projector(
            src_dist=src_dist,
            det_dist=det_dist,
            det_spacing=det_spacing,
            det_count=det_count,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            n_angles=n_angles,
            scale=scale,
            device=device,
        )

        self.radon = radon
        self.scale = float(backend_scale)
        self.fbp_window = fbp_window

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.radon.forward(x)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.radon.backward(y)

    @override
    def A_vjp(self, x: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A_adjoint(v, **kwargs)

    def A_dagger(
        self,
        y: torch.Tensor,
        filter_name: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not hasattr(self.radon, "filter_sinogram"):
            raise NotImplementedError("The backend does not implement filter_sinogram.")

        filter_name = self.fbp_window if filter_name is None else filter_name
        y_filt = self.radon.filter_sinogram(y, filter_name=filter_name)
        return self.radon.backward(y_filt)


class MultiScaleTomography(nn.Module):
    """
    Container of tomography operators at multiple scales.
    """

    def __init__(
        self,
        *,
        scales: Sequence[int] = (1, 2, 4),
        src_dist: float = 595.0,
        det_dist: float = 490.0,
        det_spacing: float = 1.0,
        det_count: int = 1024,
        dx: float = 0.8,
        dy: float = 0.8,
        nx: int = 512,
        ny: int = 512,
        n_angles: int = 60,
        scale: float = 1.0,
        device: torch.device | str = "cuda",
        fbp_window: str = "hann",
        noise_model=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.base_physics = [
                Tomography(
                    src_dist=src_dist,
                    det_dist=det_dist,
                    det_spacing=det_spacing,
                    det_count=det_count,
                    dx=dx * s,
                    dy=dy * s,
                    nx=nx // s,
                    ny=ny // s,
                    n_angles=n_angles,
                    scale=scale / (s**2),
                    device=device,
                    fbp_window=fbp_window,
                    noise_model=noise_model,
                    **kwargs,
                )
                for s in scales
            ]


        self._left_composition = None

    def __len__(self) -> int:
        return len(self.base_physics)

    @property
    def n_levels(self) -> int:
        return len(self.base_physics)

    def compose_left(self, physics: LinearPhysics) -> MultiScaleTomography:
        self._left_composition = physics
        return self

    def clear_composition(self) -> MultiScaleTomography:
        self._left_composition = None
        return self

    def get_base_physics(self, level: int = 0) -> Tomography:
        return self.base_physics[level]

    def get_physics(self, level: int = 0, composed: bool = True) -> LinearPhysics:
        physics = self.base_physics[level]
        if composed and self._left_composition is not None:
            physics = self._left_composition * physics
        return physics





