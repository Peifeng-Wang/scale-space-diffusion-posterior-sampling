from __future__ import annotations

import numpy as np
import torch

import deepinv as dinv
from typing import Any


def load_model(path: str, model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def load_drunets(configs: dict, device: str = "cuda") -> list[torch.nn.Module]:
    models = []

    for name, cfg in configs.items():
        print(name)

        print(cfg)
        weights_path = cfg["pretrained"] if "pretrained" in cfg else None

        cfg = {**cfg, "pretrained": None}

        model = dinv.models.DRUNet(**cfg)
        model = dinv.models.EquivariantDenoiser(model, random=True)

        if weights_path is not None:
            print('in loading')
            model = load_model(weights_path, model, device)

        models.append(model.to(device))

    return models


def create_multiscale_prior(
        model_type: str = 'drunet',
        configs: dict = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:

        if model_type == 'drunet':
            denoisers = load_drunets(configs, device)

        priors = []
        for denoiser in denoisers:
            priors.append(FirePrior(denoiser=denoiser))            

        return priors