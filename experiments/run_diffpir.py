import random
import sys
from pathlib import Path

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import numpy as np
import torch

from src.physics.tomography import Tomography, MultiScaleTomography, MultiScaleSinogram, MultiScaleTensor, WeightedMultiScalePhysics
from src.optim.sampler import DiffPIRSampler
from src.utils.load import load_unet_diff
from src.utils.dataloaders import get_att_ct_dataloaders
from src.utils.load import compose_cfg, to_plain_dict

from src.optim.multiscale_schedule import MultiScaleSchedule


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
    set_seed(97)

    test_dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=config["dataloader"]["loader"]["batch_size"],
        shuffle=False,
    )

    models = load_unet_diff(config_models)
    # first one in the list: 512x512
    # second one in the list: 256x256
    # third one in the list: 128x128

    sampler = DiffPIRSampler(**config["experiment"]["diffusion"], device=device)
    
    I0 = 1e6 #just for some test #config["experiment"]["params"]["intensity"]
    noise_model = dinv.physics.LogPoissonNoise(N0=I0, mu=1.0, rng=None)
    data_fidelity = dinv.optim.L2() 
    physics = Tomography(n_view=60, noise_model=noise_model)

    multiscale_physics = MultiScaleTomography(scales=(1, 2, 4), n_view=60, noise_model=noise_model, device=device)

    # ----------NEW: For integration compatibility test----------
    schedule = MultiScaleSchedule.from_block_spec(
        T=config["experiment"]["diffusion"]["num_timesteps"],
        levels=[2, 1, 0],
        transition_ts=[100, 50],
    )

    # Quick check: are the levels in the schedule compatible with the number of models and physics levels you have?
    assert max([2, 1, 0]) < len(models)
    assert max([2, 1, 0]) < multiscale_physics.n_levels

    # Quick check: how many transitions will happen in your run?
    steps = list(schedule.iter_reverse(t_start=199))
    print("transitions:", sum(s.is_transition for s in steps))
    print("first 10 steps:", steps[:10])
    # -----------------------------------------------------------

    with torch.no_grad():
        for x in test_dataloader:

            x = x.to(device)

            gt = x.clone()  # ready to plot the ground truth image

            # noiseless line integrals
            # in deepinv, calling tomography like this when a noise model
            # is set give you a 'real measurement'
            b = physics(x).clamp_min(0)

            multiscale_b = MultiScaleSinogram(b, scales=(1, 2, 4))

            x_fbp = physics.A_dagger(b).clamp(0, 0.04825).detach()
            #plot(x_fbp, 0, vmax=0.04825)   # uncomment to visualize the FBP reconstruction

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

            b_tilde = MultiScaleTensor(b_tilde_levels)

            weighted_multiscale_physics = WeightedMultiScalePhysics(
                base_physics=multiscale_physics,
                wsqrt_levels=wsqrt_levels,
            )


            x_fbp_norm = x_fbp / 0.04825


            x0 = sampler.sample(y=b_tilde,
                                physics=weighted_multiscale_physics,
                                data_fidelity=data_fidelity,
                                lambda_=1e-3,
                                model=models,
                                levels=[2, 1, 0],
                                transition_ts= [70, 30],
                                x_init=x_fbp_norm, t_start=199, noise=None, gt=gt, num_ddim_steps=50, zeta=0.4)
            break  # just one batch for testing

            #plot(xk, vmax=0.048)


if __name__=='__main__':
    main()