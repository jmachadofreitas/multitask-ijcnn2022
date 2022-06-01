from abc import ABCMeta, abstractmethod

import torch.distributions as td

from .shared import *


class ProbabilisticModule(nn.Module, metaclass=ABCMeta):
    """
    Abstract Base Class for a parameterized distribution module

    Joins torch.nn.Module and torch.distribution.Distribution

    Args:
        block [nn.module]: torch nn.Module
        input_dim: ...
        output_dim: ...
    """

    def __init__(
            self,
            block: nn.Module,
            input_dim: int,
            output_dim: int,
    ):
        super().__init__()
        self.block = block
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor):
        """ Outputs distributions parameters """
        raise NotImplementedError

    @abstractmethod
    def distribution(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def point_and_dist(self, x: Tensor):
        """ Outputs unormalized score estimate (e.g. mean, logits) and respective distribution """
        raise NotImplementedError

    @abstractmethod
    def dsample(self, param: Tensor, *params: Tensor):
        """ Differentiable (approximate) sample """
        raise NotImplementedError

    def sample(self, x: Tensor, *args, **kwargs):
        """ Distribution class wrapper """
        dist = self.distribution(x)
        return dist.sample(*args, **kwargs)

    def log_prob(self, x: Tensor, value: Tensor):
        """ Distribution class wrapper """
        dist = self.distribution(x)
        return dist.log_prob(value)

    def sample_with_log_prob(self, x: Tensor):
        """ Distribution class wrapper """
        dist = self.distribution(x)
        sample = dist.sample()
        return sample, dist.log_prob(sample)

    __repr__ = simple_repr

    def extra_repr(self) -> str:
        repr_str = "\n\tblock=" + repr(self.block)
        repr_str += f", input_dim={self.input_dim}, output_dim={self.output_dim}"
        return repr_str


class MultivariateNormalDiag(ProbabilisticModule):

    def __init__(
            self,
            block: nn.Module,
            input_dim: int,
            output_dim: int,

    ):
        """ Multivariate Normal distribution with diagonal covariance """
        super().__init__(block, input_dim, output_dim)

        # Adapters
        self.loc = nn.Linear(input_dim, output_dim)
        self.logscale = nn.Linear(input_dim, output_dim)

        # Init
        self.loc.apply(lambda p: init_weights(p, nonlinearity="linear"))
        self.logscale.apply(lambda p: init_weights(p, nonlinearity="linear"))

    def forward(self, x: Tensor):
        """ Get distribution parameters """
        h = self.block(x)
        loc, logscale = self.loc(h), self.logscale(h)
        return loc, logscale

    def _stddev(self, logscale):
        # Alt: return F.softplus(logscale - 5.)
        return torch.exp(.5 * logscale)

    def distribution(self, x):
        # Alt: mvn = td.MultivariateNormal(mean, scale_tril=torch.diag(stddev**2))
        mean, logscale = self(x)
        stddev = self._stddev(logscale)
        mvn = D.Independent(D.Normal(mean, stddev), 1)
        return mvn

    def point_and_dist(self, x):
        dist = self.distribution(x)
        return dist.mean, dist

    def dsample(self, mean, logscale_diag):
        """ Reparameterization Trick """
        return torch.randn_like(mean).mul_(self._stddev(logscale_diag)).add_(mean)
