from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F


def simple_repr(obj):
    name = obj.__class__.__name__
    extra_repr = obj.extra_repr()
    return name + "(" + extra_repr + ")"


def make_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        activation_fn = nn.LeakyReLU(0.2)
    else:
        raise NotImplementedError
    return activation_fn


@torch.no_grad()
def init_weights(module: nn.Module, nonlinearity="relu"):
    classname = module.__class__.__name__
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)
        if module.bias is not None:
            torch.nn.init.normal_(module.bias.data, std=0.01)
    elif isinstance(module, torch.nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)
        if module.bias is not None:
            torch.nn.init.normal_(module.bias.data, std=0.01)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.constant_(module.bias, 0.0)
    else:
        pass
