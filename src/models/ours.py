from typing import Sequence, Callable
from collections import OrderedDict


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule
import torchmetrics as tm

from .mixin import ModelMixin
from ..utils import *


def kl_mvn_diag_mvn_diag(mean0, logvar0, mean1, logvar1):
    return 0.5 * torch.sum(
        torch.exp(logvar0) / torch.exp(logvar1)
        + (mean1 - mean0).pow(2) / torch.exp(logvar1)
        + logvar1 - logvar0 - 1,
        dim=-1
    )


class Ours(ModelMixin, LightningModule):
    """ Multi-task Learning """

    def __init__(
            self,
            econfig: ExperimentConfig,
            dconfig: DatasetConfig,
    ):
        assert econfig.beta >= 0
        super().__init__()
        self.save_hyperparameters()

        self.econfig = econfig
        self.dconfig = dconfig

        # Model
        self.encoder = self.init_encoder(
            dconfig.input_shape,
            econfig.enc_hidden_dims,
            econfig.latent_dim,
            econfig.nonlinearity,
        )

        self.predictors = self.init_predictors(
            econfig.latent_dim,
            econfig.pred_hidden_dims,
            dconfig.target_dims
        )
        self.beta = econfig.beta
        self.lr = econfig.lr
        self.latent_dim = econfig.latent_dim
        self.with_filter = econfig.with_filter
        self.with_weights = econfig.with_weights
        self.num_tasks = dconfig.num_tasks

        if self.with_filter:
            self.filter: nn.ModuleDict = self.init_filter(self.latent_dim, self.num_tasks)

        if self.with_weights:
            self.weights: nn.ParameterList = self.init_task_weights(self.num_tasks)

        # Losses
        self.prediction_losses = self.init_prediction_losses(
            dconfig.target_types
        )

        # Place modular metrics on correct device
        self.train_metrics = self.init_metrics(dconfig.target_dims, dconfig.target_types)
        self.val_metrics = self.init_metrics(dconfig.target_dims, dconfig.target_types)
        self.test_metrics = self.init_metrics(dconfig.target_dims, dconfig.target_types)

        self.metrics = dict(
            train=self.train_metrics,
            val=self.val_metrics,
            test=self.test_metrics,
        )

    def forward(self, x):
        """ Returns inputs to task-specific networks """
        z = self.encoder(x)
        if self.with_filter:
            ws = [self.sample(z, logvar) for logvar in self.filters["n_logvars"]]
        else:
            ws = [z for _ in range(self.num_tasks)]
        return z, ws

    def _sample(self, loc, logscale):
        """ w_j sample """
        stddev = torch.exp(0.5 * logscale)
        noise = stddev * torch.randn_like(loc)
        return loc + noise

    def _encode(self, x):
        z = self.encoder(x)
        if self.with_filter:
            ws = [self._sample(z, logvar) for logvar in self.filter["n_logvars"]]
        else:
            ws = [z for _ in range(self.num_tasks)]
        return z, ws

    def _predict(self, ws):
        logits = [self.predictors[k](w) for k, w in enumerate(ws)]
        return logits

    def _filter_loss(self, z):
        kldivs = list()
        n_logvars, q_means, q_logvars = self.filter["n_logvars"], self.filter["q_means"], self.filter["q_logvars"]
        for n_logvar, q_mean, q_logvar in zip(n_logvars, q_means, q_logvars):
            kldiv = kl_mvn_diag_mvn_diag(z, n_logvar, q_mean, q_logvar).sum()
            kldivs.append(kldiv)
        logs = {f"kl-{k}": kldiv for k, kldiv in enumerate(kldivs)}
        return sum(kldivs), logs

    def _step(self, batch, stage=None):
        x, targets = batch

        # Inference
        z, ws = self._encode(x)
        logits = self._predict(ws)

        # Losses
        pred_loss, plogs = self._predictions_loss(logits, targets)
        kldiv, klogs = self._filter_loss(z) if hasattr(self, "filter") else (0., {})
        weights_loss, wlogs = self._weights_loss() if hasattr(self, "weights") else (0., {})
        loss = pred_loss + weights_loss + self.beta * kldiv

        with torch.no_grad():
            for k in range(self.num_tasks):
                self.metrics[stage][k].update(logits[k], targets[:, k])

            logs = OrderedDict(
                loss=loss,
                pred_loss=pred_loss,
                kldiv=kldiv,
                avg_z=z.mean(),
                beta=self.beta,
                **plogs,
                **klogs,
                **wlogs
            )

        return loss, logs
