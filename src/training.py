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
from models import SimCLR


# PyTorch Lightning Module
class SimCLRModule(pl.LightningModule):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        self.simclr = SimCLR(config)
        self.config = config

        self.save_hyperparameters(config)
    
    def training_step(self, batch, batch_idx):
        x1, x2, lengths = batch
        loss = self.simclr(x1, x2, lengths)
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.trainer.val_dataloaders.dataset.eval = True

    def on_validation_end(self):
        self.trainer.val_dataloaders.dataset.eval = False
    
    def validation_step(self, batch, batch_idx):
        x1, x2, lengths = batch
        val_loss = self.simclr(x1, x2, lengths)
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
    model = SimCLRModule(config)
    
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
