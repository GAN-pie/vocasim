#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, List, Dict, Union
from collections import namedtuple

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio.models

from data_utils import AudioDataset, padded_batch_collate_fn

# Losses and Training Utilities
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
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
        x = torch.mean(x, dim=1)
        return self.proj(x)

ModelConfig = namedtuple(
    'ModelConfig',
    field_names=[
        'batch_size', 'learning_rate', 'weight_decay', 'epochs',
        'warmup_steps', 'temperature',
        'input_dim', 'n_heads', 'ffn_dim', 'n_layers', 'kernel_size'
    ],
    rename=False,
    defaults=[8, 1e-5, 1e-5, 100, 5, 0.1, 80, 4, 64, 2, 31]
)

# PyTorch Lightning Module
class SimCLRAudioModel(pl.LightningModule):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            self.config = ModelConfig()._asdict()
            print(self.config)
        else:
            self.config = ModelConfig()._asdict() | config

        self.encoder = torchaudio.models.Conformer(
            input_dim=self.config['input_dim'],
            num_heads=self.config['n_heads'],
            ffn_dim=self.config['ffn_dim'],
            num_layers=self.config['n_layers'],
            depthwise_conv_kernel_size=self.config['kernel_size']   # Original paper shows similar perf. with kernel size from 7 until 32
        )
        self.projection = ProjectionModule(self.config['input_dim'], self.config['ffn_dim'])
        self.loss = NTXentLoss(self.config['temperature'])

        self.save_hyperparameters(self.config)
    
    def training_step(self, batch, batch_idx):
        aug1, aug2, lengths = batch
        z1, _ = self.encoder(aug1, lengths)
        z2, _ = self.encoder(aug2, lengths)
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        loss = self.loss(h1, h2).mean()
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.trainer.val_dataloaders.dataset.eval = True

    def on_validation_end(self):
        self.trainer.val_dataloaders.dataset.eval = False
    
    def validation_step(self, batch, batch_idx):
        aug1, aug2, lengths = batch
        z1, _ = self.encoder(aug1, lengths)
        z2, _  = self.encoder(aug2, lengths)
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        val_loss = self.loss(h1, h2).mean()
        self.log('val_loss', val_loss.detach(), on_epoch=True, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        cosine_warmup = OneCycleLR(
            optimizer,
            9.5e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.07,
            anneal_strategy='cos',
            three_phase=False
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': cosine_warmup,
                'interval': 'step',
            }
        }

def train(config: Dict):
    # Create Datasets
    train_dataset = AudioDataset(config['train_data'], config)
    val_dataset = AudioDataset(config['val_data'], config)
    
    # Instanciating DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=padded_batch_collate_fn,
        shuffle=True,
        num_workers=16
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=padded_batch_collate_fn,
        shuffle=False,
        num_workers=16
    )
    
    # Create Lightning Trainer
    model = SimCLRAudioModel(config)
    
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # precision='16-mixed',
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss', 
                mode='min', 
                save_top_k=3
            ),
            #EarlyStopping(
            #    monitor='val_loss', 
            #    patience=10
            #),
            LearningRateMonitor()
        ],
        #val_check_interval=0.5,
        log_every_n_steps=100
    )
    
    # Launch training
    trainer.fit(model, train_loader, val_loader)
