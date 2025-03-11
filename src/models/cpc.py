#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, List, Dict, Union, Tuple
from collections import namedtuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.models import Conformer


class ConvolutionModule(torch.nn.Module):
    """Using the convolution module from torchaudio.Conformer
    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

        def _weights_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
        self.apply(_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


def make_encoder(config):
    return ConvolutionModule(
        input_dim=config['input_dim'],
        num_channels=config['ffn_dim'],
        depthwise_kernel_size=config['kernel_size'],
    )


def make_autoregressive(config):
    return nn.GRU(
        config['ffn_dim'],
        config['hidden_dim'],
        batch_first=True,
        bias=False
    )


def make_projection(config):
    return nn.ModuleList([
        nn.Linear(
            config['hidden_dim'],
            config['ffn_dim'],
            bias=False
        )
        for _ in range(config['steps'])
    ])


class CPCInfoNCE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.g_enc = make_encoder(config)
        self.g_ar = make_autoregressive(config)
        self.Wk = make_projection(config)

        # TODO: initialize weights
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
        self.apply(_weights_init)
        # initialize autoregressive model
        for p_list in self.g_ar._all_weights:
            for p_name in p_list:
                if 'weight_hh' in p_name:
                    nn.init.orthogonal_(getattr(self.g_ar, p_name))

        # noise samples is equal to batch_size - 1
        # self.register_buffer('noise_samples', torch.tensor(config['noise_samples'], dtype=int))
        self.register_buffer('steps', torch.tensor(config['steps'], dtype=int))
        self.register_buffer('tau', torch.tensor(config['temperature']))

        self.config = config

    def _compute_density_ratio(self, z: Tensor, c_t: Tensor, k) -> Tensor:
        """
        Args:
            - z: the k future step encoded input to predict
            - c_t: context encoded representation
            - k: int, representing the considered step
        Returns:
            the scalar value of the density ratio f_k

        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        proj = self.Wk[k-1](c_t).T
        return (z @ proj).squeeze(0)

    def _compute_info_nce_loss(self, z: Tensor, c: Tensor, t: Tensor, reduce: bool = True) -> Tensor:

        B, T, C = z.size()
        K = self.steps

        steps_loss = []
        for k in range(1, K+1):
            rand_sequences_idx = torch.multinomial(
                (torch.eye(B) == 0.).float(),
                B-1,
                replacement=True
            ).to(z).int()
            rand_samples_idx = torch.multinomial(
                torch.ones(B, T),
                B-1,
                replacement=True
            ).to(z).int()
            negatives_samples = z[rand_sequences_idx, rand_samples_idx]

            batch_loss = []
            for b in range(B):
                f_k_pos = self._compute_density_ratio(z[b, t+k], c[b, t], k)

                f_k_negs = []
                for n in range(B-1):
                    f_k_negs += [
                        self._compute_density_ratio(negatives_samples[b, n], c[b, t], k)
                    ]
                f_k_negs = torch.cat(f_k_negs, dim=0)

                log_softmax = F.log_softmax(
                    torch.cat([f_k_pos, f_k_negs], dim=0), dim=0
                )

                loss = -1. * log_softmax[b]

                batch_loss += [loss]

            steps_loss += [sum(batch_loss) / B]

        if reduce:
            steps_loss = sum(steps_loss) / K

        return Tensor(steps_loss).to(z.device)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        K = self.steps
        
        max_t = T - K - 1
        
        t = torch.randint(max_t.int(), (1,))

        z = self.g_enc(x)
        c, _ = self.g_ar(z[:, :-K])
        
        return self._compute_info_nce_loss(z, c, t)

