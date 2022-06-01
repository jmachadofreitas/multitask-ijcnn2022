from typing import Optional, Sequence, Callable, List
from dataclasses import dataclass
import logging
import random
import traceback
import io

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torchvision as tv
import torch.distributions as td

from torchvision.models.alexnet import alexnet

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from src.datasets import *
from src.modules import *
from src.evaluators import *


@dataclass
class ExperimentConfig:
    seed: int
    latent_dim: int
    enc_hidden_dims: Sequence[int]
    pred_hidden_dims: Sequence[int]
    beta: float
    max_epochs: int
    lr: float = 1e-3
    with_filter: bool = True
    with_weights: bool = False
    nonlinearity: str = "relu"
    pretrained: bool = False


@dataclass
class DatasetConfig:
    input_shape: Tuple[int, ...]
    target_types: Sequence[str]
    target_dims: Sequence[int]
    target_names: Sequence[str]

    @property
    def num_tasks(self) -> int:
        return len(self.target_types)


def output_shape(module, input_shape):
    x = torch.rand(1, *input_shape)
    return module(x).shape[1:]


def init_datamodule(datamodule_config: dict):
    name = datamodule_config.pop("name")
    if name == "grouped_mnist":
        dm = GroupedMNISTDataModule(**datamodule_config)
    elif name == "grouped_fmnist":
        dm = GroupedMNISTDataModule(**datamodule_config)
    elif name == "multi_mnist":
        dm = MultiMNISTDataModule(**datamodule_config)
    elif name == "mtfl":
        dm = MTFLDataModule(**datamodule_config)
    else:
        raise ValueError(f"Unknown datamodule: '{name}'")
    dm.setup()
    return dm
