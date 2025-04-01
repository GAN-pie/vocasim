#!/usr/bin/env python3
# coding: utf-8

from typing import Optional, List, Dict, Union
from collections import namedtuple

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import grad_norm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torchaudio.models

from models import SimCLR, CPCInfoNCE
from dataset import SpectrogramDataset, simclr_batch_collate, padded_batch_collate, BaseAudioDataset


class SimCLRModule(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()

        self.simclr = SimCLR(config)

        self.spec_augment = torchaudio.transforms.SpecAugment(
            n_time_masks=config['n_time_mask'],
            time_mask_param=config['max_time_mask'],
            n_freq_masks=config['n_freq_mask'],
            freq_mask_param=config['max_freq_mask'],
            iid_masks=True,
            p=1.0,
            zero_masking=True
        )

        self.config = config
        self.save_hyperparameters(config)
    
    def training_step(self, batch, batch_idx):
        labels, x1, x2, lengths = batch
        x1 = self.spec_augment(x1).permute(0, 2, 1)
        x2 = self.spec_augment(x2).permute(0, 2, 1)
        lengths = Tensor(lengths).to(x1)
        loss, _ = self.simclr(x1, x2, lengths)
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels, x1, x2, lengths = batch
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        lengths = Tensor(lengths).to(x1)
        val_loss, val_accuracy = self.simclr(x1, x2, lengths)
        # self.log('val_loss', val_loss.detach(), on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                'val_loss': val_loss.detach(),
                'val_accuracy': val_accuracy.detach()
            },
            on_epoch=True,
            prog_bar=True
        )
        return val_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.simclr, norm_type=2)
        self.log_dict(norms)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        cosine_warmup = OneCycleLR(
            optimizer,
            self.config['lr'],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=self.config['annealing_factor'],
            three_phase=False
        )
        # reduce_lr = ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=1000,
        # )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': cosine_warmup,
                # 'scheduler': reduce_lr,
                # 'monitor': 'train_loss',
                'interval': 'step',
            }
        }


class CPCModule(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()

        self.cpc = CPCInfoNCE(config)

        self.config = config
        self.save_hyperparameters(config)
    
    def training_step(self, batch, batch_idx):
        labels, x, _= batch
        x = x.permute(0, 2, 1)
        loss = self.cpc(x)
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels, x, _= batch
        x = x.permute(0, 2, 1)
        val_loss = self.cpc(x)
        self.log('val_loss', val_loss.detach(), on_epoch=True, prog_bar=True)
        return val_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.cpc, norm_type=2)
        self.log_dict(norms)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        cosine_warmup = OneCycleLR(
            optimizer,
            self.config['lr'],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=self.config['annealing_factor'],
            three_phase=False
        )
        # reduce_lr = ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=1000,
        # )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': cosine_warmup,
                # 'scheduler': reduce_lr,
                # 'monitor': 'train_loss',
                'interval': 'step',
            }
        }


def train(train_data: str, val_data: str, config: Dict):
    train_dataset = SpectrogramDataset(train_data, config) \
            if config['n_fft'] else BaseAudioDataset(train_data, config)
    val_dataset = SpectrogramDataset(val_data, config) \
            if config['n_fft'] else BaseAudioDataset(val_data, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=simclr_batch_collate if config['model'] == 'simclr' else padded_batch_collate,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=simclr_batch_collate if config['model'] == 'simclr' else padded_batch_collate,
        shuffle=False,
        num_workers=8
    )

    model = SimCLRModule(config) if config['model'] == 'simclr' else CPCModule(config)
    
    trainer = pl.Trainer(
        # overfit_batches=0.15,
        gradient_clip_val=1.0,
        max_epochs=config['epochs'],
        # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # precision='16-mixed',
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss', 
                mode='min', 
                save_top_k=3
            ),
            # EarlyStopping(
            #     monitor='val_loss', 
            #     patience=5
            # ),
            LearningRateMonitor()
        ],
        #val_check_interval=0.5,
        log_every_n_steps=100
    )
    
    trainer.fit(model, train_loader, val_loader)
