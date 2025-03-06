#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.models import Conformer


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, reduce: bool = True):
        super().__init__()
        self.reduce = reduce
        self.register_buffer('tau', Tensor([temperature]))

    def forward(self, z1: Tensor, z2: Tensor):
        """
        Args:
            - z1: a Tensor with shape [B, C]
            - z2: a Tensor with shape [B, C]
        Returns:
            value(s) of the NTXent loss computed on every samples or averaged
            over batch if reduced is True
        """
        batch_size = z1.size(0)
        
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        z = torch.cat((z1_norm, z2_norm), dim=0)
        S = torch.exp(F.cosine_similarity(z[:, None, ...], z[None, ...], dim=2) / self.tau)

        mask = torch.eye(S.shape[0]).roll(torch.numel(S) // 2).bool()
        positive_pairs = S[mask]
        negative_pairs = torch.sum(S[~mask].view(S.shape[0], -1), dim=-1)

        pairs_loss = -1 * torch.log(positive_pairs / negative_pairs)
        loss = pairs_loss.view(-1, batch_size).sum(dim=0) / S.shape[0]

        if self.reduce:
            loss = loss.mean()

        return loss


class ProjectionModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        output_dim = output_dim or input_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim, bias=False)
        )

    def forward(self, x: torch.Tensor):
        return self.proj(x)


class SimCLR(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        self.encoder = Conformer(
            input_dim=self.config['input_dim'],
            num_heads=self.config['n_heads'],
            ffn_dim=self.config['ffn_dim'],
            num_layers=self.config['n_layers'],
            depthwise_conv_kernel_size=self.config['kernel_size'] # similar perf. with kernel size from 7 until 32 according to paper
        )

        self.projection = nn.Sequential(
            nn.Linear(self.config['input_dim'], self.config['ffn_dim']),
            nn.ReLU(),
            nn.Linear(self.config['ffn_dim'], self.config['input_dim'])
        )

        self.ntxent_loss = NTXentLoss(self.config['temperature'], reduce=True)

    def forward(self, x1: Tensor, x2: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            - x1 is a Tensor of shape [B, T, C]
            - x2 is a Tensor of shape [B, T, C]
            - lengths is a Tensor of shape [B]
        Returns:
            a Tensor containing loss value computed on input tensors
        """
        z1, _ = self.encoder(x1, lengths) # output shape is [B, T, C]
        z2, _ = self.encoder(x2, lengths) # output shape is [B, T, C]
        
        # averaging time axis
        z1 = z1.mean(1)
        z2 = z2.mean(1)
        
        h1 = self.projection(z1) # shape is now [B, C]
        h2 = self.projection(z2)

        return self.ntxent_loss(h1, h2)
