#!/usr/bin/env python3
# coding: utf-8

from collections import namedtuple
from typing import Optional, List, Dict, LiteralString, Tuple
import json
import random

import torch
from torch import nn, Tensor
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
        'noise_level', 'gain', 'max_freq_mask', 'max_time_mask'
    ],
    rename=False,
    defaults=[16000, 512, 256, 16000, 80, 0.01, 1.2, 64, 16]
)

# Wave based augmentation
class WaveAugmentation(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        if config is None:
            self.config = AudioConfig()._asdict()
        else:
            self.config = AudioConfig()._asdict() | config

        self.transforms = nn.ModuleDict({
            'noise': AddWhiteNoise(noise_level=self.config['noise_level']),
            'gain': T.Vol(gain=self.config['gain'], gain_type='amplitude')
        })

    @torch.no_grad()
    def forward(self, audio: Tensor):
        candidates = list(self.transforms.keys())
        numerus_closus = random.randint(1, len(candidates))
        selection = random.sample(candidates, numerus_closus)
        for k in selection:
            audio = self.transforms[k](audio)
        return audio

# Add noise transform
class AddWhiteNoise(nn.Module):
    def __init__(self, noise_level=0.01):
        super().__init__()
        self.noise_level = noise_level
   
    @torch.no_grad()
    def forward(self, audio:Tensor):
        noise = torch.randn_like(audio) * self.noise_level
        return audio + noise

# Spectrogram based augmentation
class SpectrogramAugmentation(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        if config is None:
            self.config = AudioConfig()._asdict()
        else:
            self.config = AudioConfig()._asdict() | config

        self.transforms = nn.Sequential(
            T.TimeMasking(time_mask_param=self.config['max_time_mask']),
            T.FrequencyMasking(freq_mask_param=self.config['max_freq_mask'])
        )

    @torch.no_grad()
    def forward(self, spectrogram: Tensor):
        return self.transforms(spectrogram)

# Data augmentation pipeline
class AudioAugmentationPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        super().__init__()

        config = AudioConfig()._asdict() | (config or {})

        self.wave_transforms = WaveAugmentation(config)
        self.spectrogram = T.Spectrogram(n_fft=config['frame_length'], power=2)
        self.spec_transforms = SpectrogramAugmentation(config)
        self.melscale = T.MelScale(
            config['n_coef'],
            config['sample_rate'],
            0,
            config['sample_rate']//2,
            n_stft=config['frame_length']//2+1
        )

    @torch.no_grad()
    def forward(self, audio: Tensor):
        audio = self.wave_transforms(audio)
        spectrogram = self.spectrogram(audio)
        spectrogram = self.spec_transforms(spectrogram)
        return self.melscale(spectrogram)


# Audio dataset main utility
class AudioDataset(Dataset):
    def __init__(
        self, 
        audio_data: LiteralString,
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
        self.eval = False
            
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        # Retrieve data index
        data_id = self.index_mapping[idx]
        datum = self.data_dict[data_id]
        
        # Uniformaize and resampling
        effects = [
            ['remix', '-'],
            ['lowpass', f"{self.config['sample_rate']//2}"],
            ['rate', f"{self.config['sample_rate']}"]
        ]
        audio, sample_rate = torchaudio.sox_effects.apply_effects_file(datum['wav'], effects=effects)

        audio = torchaudio.functional.preemphasis(audio, coeff=0.97)

        # Truncature
        if audio.size(1) > self.target_length:
            max_offset = audio.size(1) - self.target_length
            offset = np.random.randint(0, max_offset+1)
            audio = audio[:, offset:offset+self.target_length]

        # To train SimCLR requires two augmentations of a unique sample
        if not self.eval:
            aug1 = self.augmentation(audio).squeeze(0).T
            aug2 = self.augmentation(audio).squeeze(0).T
            return aug1, aug2   # shape [T,C], [T,C]
        # To eval SimCLR we give two copies of same sample
        else:
            mel_spec = self.augmentation.melscale(self.augmentation.spectrogram(audio))
            return mel_spec, mel_spec

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

def simclr_collate(batch: List, eval: bool = False):
    pass




class BaseAudioDataset(Dataset):
    def __init__(self, data: LiteralString, config: Optional[Dict] = None):
        super().__init__()

        # Required default audio condig or overiding it with config in arguments
        if config is None:
            self.confif = AudioConfig()._asdict()
        else:
            self.config = AudioConfig()._asdict() | config

        with open(data, 'r') as fd:
            self.data_dict = json.load(fd)

        self.index_map = {i: idx for i, idx in enumerate(self.data_dict.keys())}

        self.augmentation = None
        self.eval = False

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[LiteralString, Tensor]:
        # Retrieve data index
        data_id = self.index_map[idx]
        datum = self.data_dict[data_id]
        
        # Uniformaize and resampling
        effects = [
            ['remix', '-'],
            ['lowpass', f"{self.config['sample_rate']//2}"],
            ['rate', f"{self.config['sample_rate']}"]
        ]
        audio, sample_rate = torchaudio.sox_effects.apply_effects_file(datum['wav'], effects=effects)

        audio = torchaudio.functional.preemphasis(audio, coeff=0.97)
        return data_id, audio
