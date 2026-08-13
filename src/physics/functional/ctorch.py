import CTorch.utils.geometry as geometry
from CTorch.projector.projector_interface import Projector
from CTorch.reconstructor.fbpreconstructor import FBPReconstructor  as FBP
import numpy as np
from typing import Sequence

def create_ct_projectors(
    *,
    nx: int = 512,
    ny: int = 512,
    dx: float = 0.8,
    dy: float = 0.8,
    nu: int = 1024,
    du: float = 1.0,
    det_type: str = "curve",
    n_view: int = 720,
    view_angles: Sequence[float] | None = None,
    sad: float = [595.0],
    sdd: float = [1085.0],
    x_ofst: float = [0.0],
    y_ofst: float = [0.0],
    u_ofst: float = [0.0],
    x_src: float = [0.0],
    proj_algo: str = "SF",
    back_algo: str = "SF",
    fbp_window: str = "hamming",
    cutoff: float = 0.95,
    scale: float = 1.0,
) -> tuple[Projector, Projector, FBP, float]:
    if view_angles is None:
        view_angles = np.arange(0.0, -2.0 * np.pi, -2.0 * np.pi / n_view)

    geom = geometry.Geom2D(
        nx,
        ny,
        dx,
        dy,
        nu,
        n_view,
        view_angles,
        du,
        det_type,
        sad,
        sdd,
        xOfst=x_ofst,
        yOfst=y_ofst,
        uOfst=u_ofst,
        xSrc=x_src,
        fixed=True,
    )

    a_op = Projector(geom, "proj", proj_algo, "forward")
    a_adj_op = Projector(geom, "proj", back_algo, "back")
    fbp_op = FBP(geom, proj_algo, window=fbp_window, cutoff=cutoff)

    return a_op, a_adj_op, fbp_op, scale