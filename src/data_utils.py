#!/usr/bin/env python3
# coding: utf-8

from collections import namedtuple
from typing import Optional, List, Dict, Literal
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np

# Audio related configuration
AudioConfig = namedtuple(
    'AudioConfig',
    field_names=[
        'sample_rate', 'frame_length', 'frame_shift', 'target_length', 'n_coef',
        'noise_level', 'gain', 'pitch_shift', 'max_freq_mask', 'max_time_mask'
    ],
    rename=False,
    defaults=[16000, 512, 256, 16000, 80, 0.01, 1.2, 4, 10, 50]
)

# Data augmentation pipeline
class AudioAugmentationPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        super().__init__()

        config = AudioConfig()._asdict() | (config or {})

        # Waveform based transforms
        self.transforms = nn.ModuleDict({
            'noise': AddNoise(noise_level=config['noise_level']),
            'gain': torchaudio.transforms.Vol(gain=config['gain'], gain_type='amplitude'),
            'pitch_shift': torchaudio.transforms.PitchShift(
                sample_rate=config['sample_rate'],
                n_steps=config['pitch_shift']
            )
        })

        # Spectrogram based transforms
        self.masks = nn.ModuleDict({
            'time_mask': torchaudio.transforms.TimeMasking(time_mask_param=config['max_time_mask']),
            'freq_mask': torchaudio.transforms.FrequencyMasking(freq_mask_param=config['max_freq_mask'])
        })
        self.spectrogram = T.Spectrogram(n_fft=config['frame_length'], power=2)
        self.melscale_spectrogram = T.MelScale(
            config['n_coef'],
            config['sample_rate'],
            0,
            config['sample_rate']//2,
            n_stft=config['frame_length']//2+1
        )

    def forward(self, audio):
        with torch.no_grad():
            # First apply random waveform transforms
            keys = list(self.transforms.keys())
            num_transforms = np.random.randint(1, len(keys)+1)
            selected_keys = np.random.choice(keys, num_transforms, replace=False)

            for key in selected_keys:
                audio = self.transforms[key](audio)

            # Second apply spectrogram random masking
            spec = self.spectrogram(audio)
            for key in self.masks:
                spec = self.masks[key](spec)

            spec = self.melscale_spectrogram(spec)
        return spec

# Add noise transform
class AddNoise(nn.Module):
    def __init__(self, noise_level=0.01):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, x):
        with torch.no_grad():
            noise = torch.randn_like(x) * self.noise_level
        return x + noise

# Audio dataset main utility
class AudioDataset(Dataset):
    def __init__(
        self, 
        audio_data: Literal,
        config: Dict,
        augmentation_pipeline: Optional[AudioAugmentationPipeline] = None
    ):
        self.config = config | AudioConfig()._asdict()
        self.target_length = self.config['target_length']
        self.augmentation = augmentation_pipeline or AudioAugmentationPipeline()

        # Load data from json
        with open(audio_data, 'r') as fd:
            self.data_dict = json.load(fd)
        self.index_mapping = {i: idx for i, idx in enumerate(self.data_dict.keys())}
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        # Retrieve data index
        data_id = self.index_mapping[idx]
        datum = self.data_dict[data_id]
        sample_rate = int(datum['rate'])
        
        # Uniformaize and resampling
        effects = [['remix', '-']]
        if sample_rate != self.config['sample_rate']:
            effects += [
                ['lowpass', f"{self.config['sample_rate']//2}"],
                ['rate', f"{self.config['sample_rate']}"]
            ]
        audio, sample_rate = torchaudio.sox_effects.apply_effects_file(datum['wav'], effects=effects)

        # Truncature
        if audio.size(1) > self.target_length:
            max_offset = audio.size(1) - self.target_length
            offset = np.random.randint(0, max_offset+1)
            audio = audio[:, offset:offset+self.target_length]

        # SimCLR asks for two augmentations
        aug1 = self.augmentation(audio).squeeze(0).T
        aug2 = self.augmentation(audio).squeeze(0).T
        return aug1, aug2   # shape [[T,C], [T,C]]

# Custom collate function for DataLoader
def padded_batch_collate_fn(batch: List):   # batch shape: [(T, C), (T, C)]
    batch_size = len(batch)
    max_length = int(max([len(row[0]) for row in batch]))

    padded_batch_aug1 = []
    padded_batch_aug2 = []
    lengths = []

    for i in range(batch_size):
        pad_length = max_length - batch[i][0].size(0)
        lengths += [batch[i][0].size(0)]
        if pad_length:
            padded_batch_aug1 += [F.pad(batch[i][0], (0, 0, 0, pad_length))]
            padded_batch_aug2 += [F.pad(batch[i][1], (0, 0, 0, pad_length))]
        else:
            padded_batch_aug1 += [batch[i][0]]
            padded_batch_aug2 += [batch[i][1]]

    # return shape [B, T, C], [B, T, C], [B,]
    return torch.stack(padded_batch_aug1), torch.stack(padded_batch_aug2), torch.Tensor(lengths)
