from typing import Optional, Sequence, Callable, List
from dataclasses import dataclass
import logging
import random
import warnings

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


warnings.filterwarnings("ignore")


class ModelMixin(object):

    @staticmethod
    def output_shape(module, input_shape):
        x = torch.rand(1, *input_shape)
        return module(x).shape[1:]

    def init_pretrained_encoder(
            self,
            latent_dim: int
    ):
        """
        Init Pretrained AlexNet

        References:
            * AlexNet  https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        """

        alexnet_module = tv.models.alexnet(pretrained=True)
        features = alexnet_module.features
        avgpool = alexnet_module.avgpool
        pre_latent = nn.Linear(256 * 6 * 6, latent_dim)

        # Freeze layers
        for param in features.parameters():
            param.requires_grad = False
        for param in avgpool.parameters():
            param.requires_grad = False

        block = nn.Sequential(
            features,
            avgpool,
            nn.Flatten(),
            pre_latent
        )
        return block

    def init_encoder(
            self,
            input_shape: Sequence[int],
            hidden_dims: Sequence[int],
            output_dim: int,
            nonlinearity: str = "relu",
    ):

        if hidden_dims is None:
            hidden_dims = list()

        if len(input_shape) == 1:
            input_dim = input_shape[0]
            encoder = MLPBlock(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim
            )
            nonlinearity = "linear" if len(hidden_dims) == 0 else nonlinearity
            encoder.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
            return encoder
        elif len(input_shape) == 3:  # MTFL
            return self.init_pretrained_encoder(latent_dim=output_dim)
        else:
            raise NotImplementedError(input_shape)

    def init_predictor(
            self,
            input_shape: Sequence[int],
            hidden_dims: Sequence[int],
            target_dim: int,
            hidden_channels: List[int] = (64, 32),
            out_channels: int = 1,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        if isinstance(input_shape, int):
            input_dim = input_shape
            predictor = MLPBlock(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=target_dim
            )
        elif len(input_shape) == 1:
            input_dim = input_shape[0]
            predictor = MLPBlock(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=target_dim
            )
        elif len(input_shape) > 1:
            in_channels, in_width, in_height = input_shape
            conv_block = Conv2dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                activation=activation,
                batch_norm=batch_norm,
            )
            output_shape = conv_block.output_shape(input_shape)
            mlp = MLPBlock(input_dim=np.prod(output_shape), hidden_dims=hidden_dims, output_dim=target_dim)
            predictor = nn.Sequential(conv_block, nn.Flatten(), mlp)
        else:
            raise NotImplemented
        return predictor

    def init_predictors(
            self,
            input_dim,
            hidden_dims,
            target_dims) -> nn.ModuleList:
        predictors = nn.ModuleList([
            self.init_predictor(
                input_shape=(input_dim,),
                hidden_dims=hidden_dims,
                target_dim=target_dim
            )
            for target_dim in target_dims
        ])
        return predictors

    def init_filter(
            self,
            latent_dim: int,
            num_tasks: int
    ) -> nn.ModuleDict:

        n_logvars = nn.ParameterList(
            [
                nn.Parameter(0.01 * torch.randn(latent_dim), requires_grad=True)
                for _ in range(num_tasks)
            ]
        )

        q_means = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
                for _ in range(num_tasks)
            ]
        )

        q_logvars = nn.ParameterList(
            [
                nn.Parameter(0.01 * torch.randn(latent_dim), requires_grad=True)
                for _ in range(num_tasks)
            ]
        )

        return nn.ModuleDict(dict(n_logvars=n_logvars, q_means=q_means, q_logvars=q_logvars))

    def init_task_weights(
            self,
            num_tasks: int
    ) -> nn.ParameterList:
        task_weigths = nn.ParameterList(
            [
                nn.Parameter(0.01 * torch.randn(1), requires_grad=True)
                for _ in range(num_tasks)
            ]
        )
        return task_weigths

    def init_prediction_losses(
            self,
            target_types
    ):
        losses = list()
        for target_type in target_types:
            if target_type == "bin":
                losses.append(nn.BCEWithLogitsLoss(reduction="sum"))
            if target_type == "cat":
                losses.append(nn.CrossEntropyLoss(reduction="sum"))
            if target_type == "num":
                losses.append(nn.MSELoss(reduction="sum"))
        return nn.ModuleList(losses)

    def init_metrics(self, target_dims, target_types):
        metrics = list()
        for target_dim, target_type in zip(target_dims, target_types):
            if target_type == "bin":
                metrics.append(
                    tm.Accuracy(
                        num_classes=target_dim + 1,
                        multiclass=True
                    )
                )
            if target_type == "cat":
                metrics.append(
                    tm.Accuracy(
                        num_classes=target_dim
                    )
                )
            if target_type == "num":
                metrics.append(tm.MeanAbsoluteError())
        return nn.ModuleList(metrics)

    def _eval_step(self, batch, stage):
        loss, logs = self._step(batch, stage=stage)
        self.log_dict({f"{stage}/{metric_name}": value for metric_name, value in logs.items()})
        for k, m in enumerate(self.metrics[stage]):
            self.log(f"{stage}/acc-{k}", m, prog_bar=True)
        return logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, stage="train")
        self.log_dict({f"train/{k}": v for k, v in logs.items()})
        for k, m in enumerate(self.metrics["train"]):
            self.log(f"train/acc-{k}", m, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logs = self._eval_step(batch, stage="val")
        return logs

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="test")

    def configure_optimizers(self):
        params_non_frozen = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.Adam(params_non_frozen, lr=self.lr)
        return opt

    def _weights_loss(self):
        weights = list()  # log(sigma)
        for k in range(self.num_tasks):
            weight_k = .5 * self.weights[k]
            weights.append(weight_k)
        logs = {f"w-{k}": w for k, w in enumerate(weights)}
        return sum(weights), logs

    def _predictions_loss(self, logits, targets):
        plosses = list()
        for k, task_logits in enumerate(logits):
            ploss = self.prediction_losses[k](task_logits, targets[:, k])
            if hasattr(self, "weights"):
                ploss = ploss / self.weights[k].exp()
            plosses.append(ploss)
        logs = {f"ploss-{k}": ploss for k, ploss in enumerate(plosses)}
        return sum(plosses), logs
