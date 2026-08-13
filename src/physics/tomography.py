from __future__ import annotations

import numpy as np
import torch

import deepinv as dinv
from deepinv.physics import LinearPhysics
from typing_extensions import override
from typing import Any, Sequence

from src.physics.functional.ctorch import create_ct_projectors

class DiagWeightPhysics(dinv.physics.LinearPhysics):
    def __init__(self, wsqrt, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("wsqrt", wsqrt)

    def A(self, z, **kwargs):
        return self.wsqrt * z

    def A_adjoint(self, z, **kwargs):
        return self.wsqrt * z

class Tomography(LinearPhysics):
    """
    Wrap custom CT forward/back projectors into a DeepInv LinearPhysics object.
    """
    def __init__(
        self,
        *,
        nx: int = 512,
        ny: int = 512,
        dx: float = 0.8,
        dy: float = 0.8,
        nu: int = 1024,
        du: float = 1.0,
        det_type: str = "curve",
        n_view: int = 720,
        view_angles=None,
        proj_algo: str = "SF",
        back_algo: str = "SF",
        fbp_window: str = "hamming",
        scale: float = 1.0,
        cutoff: float = 0.95,
        noise_model=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,   
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_model=noise_model, **kwargs)

        # Build backend projectors.
        A_op, A_adj_op, fbp_op, backend_scale = create_ct_projectors(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            nu=nu,
            du=du,
            det_type=det_type,
            n_view=n_view,
            view_angles=view_angles,
            proj_algo=proj_algo,
            back_algo=back_algo,
            fbp_window=fbp_window,
            scale=scale,
            cutoff=cutoff
        )

        self.A_op = A_op
        self.A_adj_op = A_adj_op
        self.fbp_op = fbp_op
        self.scale = float(scale)

    def A(self, x, **kwargs):
        return self.A_op(x)

    def A_adjoint(self, y, **kwargs):
        ATy = self.scale * self.A_adj_op(y)
        return ATy
    
    @override
    def A_vjp(self, x, v):
        return self.A_adjoint(v)
    
    def A_dagger(self, y, **kwargs):
        """Run FBP reconstruction if provided."""
        if self.fbp_op is None:
            raise NotImplementedError("No FBP operator was provided.")
        return self.fbp_op(y)


def downsample_detector_bins(y: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Average only adjacent detector bins in a sinogram.

    The angle dimension is unchanged. For factor=1, the original tensor is
    returned unchanged.
    """
    if factor == 1:
        return y
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}.")
    det_count = y.shape[-1]
    if det_count % factor != 0:
        raise ValueError(
            f"detector dimension {det_count} is not divisible by factor {factor}."
        )
    return y.reshape(*y.shape[:-1], det_count // factor, factor).mean(dim=-1)


class MultiScaleSinogram:
    """
    Lightweight container for sinograms matched to each image scale.

    Level 0 keeps the fine measurement unchanged. Higher levels average only
    adjacent detector bins according to the corresponding scale factor.
    """
    def __init__(self, y_fine: torch.Tensor, scales: Sequence[int] = (1, 2, 4)) -> None:
        self.y_fine = y_fine
        self.scales = tuple(scales)
        if any(s < 1 for s in self.scales):
            raise ValueError(f"All scales must be >= 1, got {self.scales}.")
        self.measurements = [downsample_detector_bins(y_fine, s) for s in self.scales]

    @property
    def device(self):
        return self.y_fine.device

    @property
    def shape(self):
        return self.y_fine.shape

    def get_measurement(self, level: int) -> torch.Tensor:
        if level < 0 or level >= len(self.measurements):
            raise IndexError(f"level {level} out of range for {len(self.measurements)} measurements.")
        return self.measurements[level]


class MultiScaleTensor:
    """
    In charge of storing tensors at multiple scales, e.g. for measurements at each level.
    """
    def __init__(self, measurements):
        self.measurements = measurements

    @property
    def device(self):
        return self.measurements[0].device

    @property
    def shape(self):
        return self.measurements[0].shape

    def get_measurement(self, level):
        return self.measurements[level]


class WeightedMultiScalePhysics:
    def __init__(self, base_physics, wsqrt_levels):
        self.base_physics = base_physics
        self.wsqrt_levels = wsqrt_levels

    @property
    def n_levels(self):
        return self.base_physics.n_levels

    def get_physics(self, level):
        W_level = DiagWeightPhysics(wsqrt=self.wsqrt_levels[level])
        A_level = self.base_physics.get_physics(level, composed=False)
        return W_level * A_level


class MultiScaleTomography(LinearPhysics):
    """
    Wrap custom CT forward/back projectors into a DeepInv LinearPhysics object.
    """
    def __init__(
        self,
        *,
        nx: int = 512,
        ny: int = 512,
        dx: float = 0.8,
        dy: float = 0.8,
        nu: int = 1024,
        du: float = 1.0,
        det_type: str = "curve",
        n_view: int = 720,
        view_angles=None,
        proj_algo: str = "SF",
        back_algo: str = "SF",
        fbp_window: str = "hamming",
        scales: Sequence[int] = [1, 2, 4],
        cutoff: float = 0.95,
        noise_model=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_model=noise_model, **kwargs)

        self.base_physics = []

        for i, scale in enumerate(scales):
            tomography = Tomography(
                nx=nx//scale,
                ny=ny//scale,
                dx=dx*scale,
                dy=dy*scale,
                nu=nu // scale, # Changed from nu=nu
                du=du * scale,  # Changed from du=du
                det_type=det_type,
                n_view=n_view,
                view_angles=view_angles,
                proj_algo=proj_algo,
                back_algo=back_algo,
                fbp_window=fbp_window,
                scale=1.0/(scale**2), #(4**i),
                cutoff=cutoff,
                noise_model=None,   # keep noise at the outer MultiScaleTomography level
                device=device,
                dtype=dtype,
            )
            self.base_physics.append(tomography)
        
        self._left_composition = None


    def A(self, x, level=0, **kwargs):
        return self.base_physics[level].A(x, **kwargs)

    def A_adjoint(self, y, level=0, **kwargs):
        return self.base_physics[level].A_adjoint(y, **kwargs)
    
    @override
    def A_vjp(self, x, v, level=0, **kwargs):
        return self.base_physics[level].A_adjoint(v, **kwargs)
    
    def A_dagger(self, y, level=0, **kwargs):
        """Run FBP reconstruction if provided."""
        if self.base_physics[level].fbp_op is None:
            raise NotImplementedError("No FBP operator was provided.")
        return self.base_physics[level].fbp_op(y)
    
    @property
    def n_levels(self) -> int:
        return len(self.base_physics)
    
    def compose_left(self, physics: LinearPhysics):
        """
        Set a physics operator to be composed on the left of every level.

        If physics = Wsqrt, then get_physics(level) returns:
            Wsqrt * tomography[level]
        """
        self._left_composition = physics
        return self

    def clear_composition(self):
        self._left_composition = None
        return self

    def get_physics(self, level: int = 0, composed: bool = True) -> LinearPhysics:
        physics = self.base_physics[level]

        if composed and self._left_composition is not None:
            physics = self._left_composition * physics

        return physics

    def __len__(self):
        return len(self.base_physics)