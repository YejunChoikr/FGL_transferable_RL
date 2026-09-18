"""Surrogate architectures: the 139,369-parameter CNN and the 140,361 MLP.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .protocol import surrogate_cfg

_S = surrogate_cfg()
_ARCH = _S["architecture"]
_MLP = _S["mlp"]

LEAKY: float = float(_ARCH["negative_slope"])
DROPOUT: float = float(_ARCH["dropout_after_second_pool"])
OUTPUT_DIM: int = int(_ARCH["fc_widths"][-1])
BN_EPS: float = float(_ARCH["batchnorm_eps"])
BN_MOMENTUM: float = float(_ARCH["batchnorm_momentum"])

CNN_EXPECTED_PARAMS: int = int(_ARCH["expected_trainable_params"])
MLP_EXPECTED_PARAMS: int = int(_MLP["expected_trainable_params"])

#: Output-layer keys reset on transfer (protocol.surrogate.transfer).
CNN_OUTPUT_KEYS = tuple(_S["transfer"]["reset_output_keys"])


class CNN(nn.Module):
    """Convolutional surrogate mapping the 60-dim state to nine displacements.

    Channels 1-16-32-64-128 with BatchNorm and LeakyReLU after each convolution,
    MaxPool(2) after the second and fourth convolutions, Dropout(0.15) after the
    second pool, then a linear head 256-128-64-9 with a linear output.
    """

    def __init__(self, output_dim: int = OUTPUT_DIM,
                 leakyrelu_para: float = LEAKY) -> None:
        super().__init__()
        ch = [int(c) for c in _ARCH["conv_channels"]]
        k = int(_ARCH["conv_kernel"])
        pad = int(_ARCH["padding"])
        bias = bool(_ARCH["bias"])
        pool_k = int(_ARCH["pool_kernel_stride"])
        fc_w = [int(w) for w in _ARCH["fc_widths"]]

        self.block1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=k, padding=pad, bias=bias),
            nn.BatchNorm2d(ch[1], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(ch[1], ch[2], kernel_size=k, padding=pad, bias=bias),
            nn.BatchNorm2d(ch[2], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(pool_k),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=k, padding=pad, bias=bias),
            nn.BatchNorm2d(ch[3], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(ch[3], ch[4], kernel_size=k, padding=pad, bias=bias),
            nn.BatchNorm2d(ch[4], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(pool_k),
        )
        self.fc = nn.Sequential(
            nn.Linear(fc_w[0], fc_w[1]),
            nn.LeakyReLU(leakyrelu_para),
            nn.Linear(fc_w[1], fc_w[2]),
            nn.LeakyReLU(leakyrelu_para),
            nn.Linear(fc_w[2], output_dim),
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 1, 10, 6)
        x = self.block1(x)
        x = self.block2(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLP(nn.Module):
    """Parameter-matched dense surrogate 60-288-288-128-9.

    BatchNorm1d and LeakyReLU follow every hidden linear layer; Dropout(0.15)
    sits after the second hidden block.
    """

    def __init__(self, output_dim: int = OUTPUT_DIM,
                 leakyrelu_para: float = LEAKY) -> None:
        super().__init__()
        w = [int(v) for v in _MLP["widths"]]
        p = float(_MLP["dropout"])
        self.net = nn.Sequential(
            nn.Linear(w[0], w[1]),
            nn.BatchNorm1d(w[1], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.Linear(w[1], w[2]),
            nn.BatchNorm1d(w[2], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.Dropout(p),
            nn.Linear(w[2], w[3]),
            nn.BatchNorm1d(w[3], eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.LeakyReLU(leakyrelu_para),
            nn.Linear(w[3], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#: Output-layer keys of the MLP, reset on transfer.
MLP_OUTPUT_KEYS = ("net.10.weight", "net.10.bias")

ARCHITECTURES = {"cnn": CNN, "mlp": MLP}
EXPECTED_PARAMS = {"cnn": CNN_EXPECTED_PARAMS, "mlp": MLP_EXPECTED_PARAMS}
OUTPUT_KEYS = {"cnn": CNN_OUTPUT_KEYS, "mlp": MLP_OUTPUT_KEYS}


def build_model(arch: str, **kwargs) -> nn.Module:
    """Instantiate the architecture named ``arch`` ("cnn" or "mlp")."""
    try:
        cls = ARCHITECTURES[arch]
    except KeyError as exc:
        raise KeyError("unknown architecture %r; known: %s"
                       % (arch, sorted(ARCHITECTURES))) from exc
    return cls(**kwargs)


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def assert_param_count(model: nn.Module, arch: str) -> int:
    """Raise unless the trainable parameter count matches the protocol."""
    n = count_params(model)
    expected = EXPECTED_PARAMS[arch]
    if n != expected:
        raise AssertionError("%s has %d trainable parameters, protocol "
                             "requires %d" % (arch, n, expected))
    return n


def transfer_copy_keys(arch: str, state_dict: dict) -> list:
    """Keys copied from the source model on transfer.

    Every parameter and buffer except the output layer and the BatchNorm
    running statistics, i.e. all convolution and hidden FC parameters plus the
    BatchNorm affine parameters (protocol.surrogate.transfer.copy).
    """
    out = set(OUTPUT_KEYS[arch])
    reset_suffixes = tuple(_S["transfer"]["reset_bn_buffers"])
    keys = []
    for k in state_dict:
        if k in out:
            continue
        if k.rsplit(".", 1)[-1] in reset_suffixes:
            continue
        keys.append(k)
    return keys


def reset_bn_running_stats(model: nn.Module) -> int:
    """Reset BatchNorm running mean/var/counter, keeping affine weight/bias."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            n += 1
    return n


__all__ = ["CNN", "MLP", "ARCHITECTURES", "EXPECTED_PARAMS", "OUTPUT_KEYS",
           "CNN_EXPECTED_PARAMS", "MLP_EXPECTED_PARAMS", "CNN_OUTPUT_KEYS",
           "MLP_OUTPUT_KEYS", "build_model", "count_params",
           "assert_param_count", "transfer_copy_keys", "reset_bn_running_stats",
           "LEAKY", "DROPOUT", "OUTPUT_DIM"]
