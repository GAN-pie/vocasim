#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, List, Dict, Union, Tuple
from collections import namedtuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import resnet18
from torchaudio.models import Conformer

from losses import NTXentLoss


ModelConfig = namedtuple(
    'ModelConfig',
    field_names=[
        'batch_size', 'learning_rate', 'weight_decay', 'epochs',
        'warmup_steps', 'temperature',
        'input_dim', 'n_heads', 'ffn_dim', 'n_layers', 'kernel_size'
    ],
    rename=False,
    defaults=[8, 1e-5, 1e-5, 1, 5, 0.1, 80, 4, 64, 2, 31]
)

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
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            self.config = ModelConfig()._asdict()
            print(self.config)
        else:
            self.config = ModelConfig()._asdict() | config

        self.encoder = Conformer(
            input_dim=self.config['input_dim'],
            num_heads=self.config['n_heads'],
            ffn_dim=self.config['ffn_dim'],
            num_layers=self.config['n_layers'],
            depthwise_conv_kernel_size=self.config['kernel_size']   # Original paper shows similar perf. with kernel size from 7 until 32
        )
        self.projection = nn.Sequential(
            nn.Linear(self.config['input_dim'], self.config['ffn_dim']),
            nn.ReLU(),
            nn.Linear(self.config['ffn_dim'], self.config['input_dim'])
        )
        self.ntxent_loss = NTXentLoss(self.config['temperature'], reduce=True)

    def forward(self, x1: Tensor, x2: Tensor, lengths: Tensor) -> Tensor:
        z1, _ = self.encoder(x1, lengths)
        z2, _ = self.encoder(x2, lengths)
        h1 = self.projection(z1.mean(1))
        h2 = self.projection(z2.mean(1))
        return self.ntxent_loss(h1, h2)


class ContrastivePredictiveCoding(nn.Module):
    def __init__(self, steps: int = 4):
        super().__init__()

        self.steps = steps

        self.g_enc = resnet18(num_classes=64)
        self.g_ar = nn.GRU(64, 128)
        self.Wk = nn.ModuleList(
            (nn.Linear(64, 64) for _ in range(self.steps))
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.g_enc(x)
        c = torch.zeros(x.size(0), self.steps, x.size(-1))
        return z, c
        

