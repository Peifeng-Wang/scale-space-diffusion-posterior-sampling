import random
import sys
from pathlib import Path

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import numpy as np
import torch

from physics.tomography import Tomography, DiagWeightPhysics, MultiScaleTomography
from src.optim.sampler import DDPMSampler
from src.utils.load import load_unet_diff
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


    models = load_unet_diff(config_models)
    # first one in the list: 512x512
    # second one in the list: 256x256
    # third one in the list: 128x128

    # let's generates in 256x256:
    model = models[1]

    diffusion = DDPMSampler(**config["experiment"]["diffusion"], device=device)
    diffusion.sample(model, predict_x0 = False,
                            image_size=256, batch_size=1,
                            channels=1)






if __name__=='__main__':
    main()