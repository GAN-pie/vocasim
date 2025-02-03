#!/usr/bin/env python3
# coding: utf-8

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, reduce: bool = True):
        super().__init__()
        self.reduce = reduce
        self.register_buffer('tau', Tensor([temperature]))

    def forward(self, z1: Tensor, z2: Tensor):
        batch_size = z1.size(0)
        
        # Normalisation
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
