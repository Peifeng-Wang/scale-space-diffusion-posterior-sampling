import random
import sys
from pathlib import Path

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import numpy as np
import torch

from physics.tomography import Tomography, DiagWeightPhysics, MultiScaleTomography
from src.utils.dataloaders import get_att_ct_dataloaders
from src.utils.load import compose_cfg, to_plain_dict
from src.utils.plot import plot

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    set_seed(98)

    test_dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=config["dataloader"]["loader"]["batch_size"],
        shuffle=True,
    )

    I0 = 1e8 #just for some test #config["experiment"]["params"]["intensity"]
    noise_model = dinv.physics.LogPoissonNoise(N0=I0, mu=1.0, rng=None)
    data_fidelity = dinv.optim.L2() 
    multiscale_tomography = MultiScaleTomography(n_view=60, noise_model=noise_model)
    
    # retrieve physics before compose them
    fine_physics = multiscale_tomography.get_physics(level=0)
    coarse_physics = multiscale_tomography.get_physics(level=1)


    with torch.no_grad():
        for x in test_dataloader:

            x = x.to(device)

            # noiseless line integrals
            # in deepinv, calling tomography like this when a noise model
            # is set give you a 'real measurement'
            b = fine_physics(x).clamp_min(0)

            x_fbp = fine_physics.A_dagger(b).clamp_min(0).detach()
            plot(x_fbp, 0, vmax=0.04825)

            x_fbp_reduced = coarse_physics.A_dagger(b).clamp_min(0).detach()
            plot(x_fbp_reduced, 0, vmax=0.04825)

        
            # oracle weights for testing only
            w = I0 * torch.exp(-b)
            Wsqrt = DiagWeightPhysics(wsqrt=w.sqrt())

            # compose weighted operator to solve 1/2||W^(1/2)(Ax - b)||^2_2
            # multiscale_tomography.compose_left(Wsqrt)
            # if so, do not forget to update b_tilde = Wsqrt.A(b) 
            multiscale_tomography.compose_left(Wsqrt)
            weighted_coarse_physics = multiscale_tomography.get_physics(level=1)
            b_tilde = Wsqrt.A(b)

            xk = torch.zeros_like(x_fbp_reduced)
            xk = data_fidelity.prox(xk, #xk is both initialization of the algo and proxed variable in that case
                                    b_tilde,
                                    weighted_coarse_physics,
                                    gamma=1e-3)
            
            plot(xk, vmax=0.048)


if __name__=='__main__':
    main()