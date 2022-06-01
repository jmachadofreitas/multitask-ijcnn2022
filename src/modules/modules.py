from typing import Optional, Sequence

from .shared import *


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(),
                 batchnorm=False,
                 dropout=False,
                 dropout_rate=0.2,
                 bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = (output_dim,)
        self.hidden_dims = hidden_dims
        self.activation_fn = nn.ReLU(inplace=True)

        # MLP Block
        layers = list()
        prev_dim = input_dim
        for dim in self.hidden_dims:
            if dropout:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(prev_dim, dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(self.activation_fn)
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

        # Initialization
        if len(hidden_dims) == 0:
            self.apply(lambda p: init_weights(p, nonlinearity="linear"))
        else:
            self.apply(lambda p: init_weights(p, nonlinearity="relu"))

    def forward(self, x):
        return self.net(x).squeeze()


class Conv2dBlock(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: Sequence[int],
            kernel_size: int = 3,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        activation_fn = make_activation_fn(activation)

        # Convolution block
        layers = list()
        prev_dim = self.in_channels
        kwargs = dict(
            kernel_size=3,
            stride=1,
            padding_mode="reflect"
        )
        for next_dim in self.hidden_channels:
            layers.append(nn.Conv2d(prev_dim, next_dim, **kwargs))
            if batch_norm:
                layers.append(nn.BatchNorm2d(next_dim))
            layers.append(activation_fn)
            prev_dim = next_dim
        self.conv_net = nn.Sequential(*layers)

        # Initialization
        if len(hidden_channels) == 0:
            self.apply(lambda p: init_weights(p, nonlinearity="linear"))
        else:
            self.apply(lambda p: init_weights(p, nonlinearity=activation))

    def output_shape(self, input_shape):
        x = torch.rand(1, *input_shape)
        x = self.conv_net(x)
        return x.shape[1:]

    def forward(self, x):
        return self.conv_net(x)


class ResConv2dBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def conv3x3(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
