from typing import Sequence
from collections import OrderedDict
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule

from .mixin import ModelMixin
from ..modules import MultivariateNormalDiag
from ..utils import ExperimentConfig, DatasetConfig


def kl_mvn_diag_std(mean, logvar):
    """
    Between factorized normal distribution N(mean, sigma * I) and standard distribution N(0, I)

    References:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mean.pow(2) - 1 - logvar, dim=1)


class MTVIB(ModelMixin, LightningModule):

    def __init__(
            self,
            econfig: ExperimentConfig,
            dconfig: DatasetConfig,
    ):
        """
        Multitask Variational Information Bottleneck

        References:
            * https://arxiv.org/abs/2007.00339
        """
        assert econfig.beta >= 0
        super().__init__()
        self.save_hyperparameters()

        self.econfig = econfig
        self.dconfig = dconfig

        # Model
        self.encoder = MultivariateNormalDiag(
            block=self.init_encoder(
                dconfig.input_shape,
                econfig.enc_hidden_dims,
                econfig.latent_dim,
                econfig.nonlinearity,
            ),
            input_dim=econfig.latent_dim,
            output_dim=econfig.latent_dim
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
        """ Returns input to task-specific networks """
        return self.encoder(x)

    def _encode(self, x):
        """ Stochastic encoder """
        loc, logscale = self.encoder(x)
        if self.training:  # Stochastic encoder
            sample = self.encoder.dsample(loc, logscale)
            return sample, loc, logscale
        else:  # Deterministic encoder
            return loc, loc, logscale

    def _predict(self, z):
        logits = [self.predictors[k](z) for k in range(self.num_tasks)]
        return logits

    def _step(self, batch, stage=None):
        x, targets = batch

        sample, mean, logvar = self._encode(x)
        logits = self._predict(sample)

        # Losses
        pred_loss, plogs = self._predictions_loss(logits, targets)
        kldiv = kl_mvn_diag_std(mean, logvar).sum()
        weights_loss, wlogs = self._weights_loss() if hasattr(self, "weights") else (0., {})
        loss = pred_loss + weights_loss + self.beta * kldiv

        with torch.no_grad():
            for k in range(self.num_tasks):
                self.metrics[stage][k].update(logits[k], targets[:, k])

            logs = OrderedDict(
                loss=loss,
                pred_loss=pred_loss,
                kldiv=kldiv,
                avg_z=sample.mean(),
                beta=self.beta,
                **plogs,
                **wlogs
            )
        return loss, logs
