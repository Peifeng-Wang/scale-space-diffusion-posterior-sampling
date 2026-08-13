import random
import sys
from pathlib import Path

# Make the repository root importable regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepinv as dinv
import numpy as np
import torch

from src.physics.tomography import Tomography
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(97)

    test_dataloader = get_att_ct_dataloaders(
        root_dir=config["dataloader"]["dataset"]["root"],
        patient_list=config["dataloader"]["loader"]["test_list"],
        batch_size=config["dataloader"]["loader"]["batch_size"],
        shuffle=False,
    )

    I0 = 1e6
    noise_model = dinv.physics.LogPoissonNoise(N0=I0, mu=1.0, rng=None)
    physics = Tomography(n_view=60, noise_model=noise_model)

    output_dir = project_root / "outputs" / "fbp"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x in test_dataloader:
            x = x.to(device)
            b = physics(x).clamp_min(0)
            x_fbp = physics.A_dagger(b).clamp(0, 0.04825).detach()

            plot(x, 0, name=str(output_dir / "ground_truth.png"), vmax=0.04825)
            plot(x_fbp, 0, name=str(output_dir / "fbp_reconstruction.png"), vmax=0.04825)

            print(f"Saved ground truth to: {output_dir / 'ground_truth.png'}")
            print(f"Saved FBP reconstruction to: {output_dir / 'fbp_reconstruction.png'}")
            break


if __name__ == "__main__":
    main()
