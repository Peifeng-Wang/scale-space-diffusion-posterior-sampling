from __future__ import annotations


import yaml
import torch
import pandas as pd
from typing import List
import os
import numpy as np
import torch

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import deepinv as dinv
from src.models.unet import UNetModel

def compose_cfg(config_dir: str | Path, config_name: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """Compose a Hydra config programmatically.

    Example:
        cfg = compose_cfg(
            config_dir="conf",
            overrides=["paths=local", "experiment=sparse_view/60"]
        )
    """
    overrides = overrides or []
    config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def to_plain_dict(node: Any) -> Any:
    """Convert an OmegaConf node into plain Python dict/list/scalars."""
    return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)


def print_resolved_cfg(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))


def load_model(path: str, model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_unet_diff(configs: dict, device: str = "cuda") -> list[torch.nn.Module]:
    models = []

    for name, cfg in configs.items():
        print(name)

        # print(cfg)
        weights_path = cfg["pretrained"] if "pretrained" in cfg else None

        cfg = {**cfg, "pretrained": None}

        model = UNetModel(**cfg)

        if weights_path is not None:
            print('in loading')
            model = load_model(weights_path, model, device)

        models.append(model.to(device))

    return models


