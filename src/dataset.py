#:/usr/bin/env python3
# coding: utf-8

import os
from os import path
import json
import random
from typing import List, Tuple, Optional, Dict
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import SpecAugment, Spectrogram, MelScale


def _truncate(x: Tensor, target_size: int) -> Tensor:
    assert x.dim() == 2, f'error expecting 2d Tensor, got {x.dim()}'
    channels, steps = x.size()
    assert steps > target_size, f'error steps less than {target_size}, got {steps}'
    max_offset = steps - target_size
    offset = random.randint(0, max_offset)
    x = torch.narrow(x, -1, offset, target_size)
    return x

def _apply_padding(x: Tensor, target_size: int) -> Tensor:
    assert x.dim() == 3, f'error expecting 3d Tensor, got {x.dim()}'
    batch_size, channels, steps = x.size()
    assert target_size >= steps, f'cannot pad tensor to {target_size} with size {steps}'
    pad_len = target_size - steps
    x = F.pad(x, (0, pad_len))
    return x


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        data: str,
        config: Dict,
    ):
        assert path.exists(data), f'ERROR, cannot find {data}'
        super().__init__()
        with open(data, 'r') as fd:
            self.data_dict = json.load(fd)
        self.index_map = {i: k for i, k in enumerate(self.data_dict)}
        self.sr = config['sample_rate']
        self.target_size = config['target_size']

    def __len__(self):
        return len(self.data_dict)

    def _load_audio(self, audio_path: str):
        effects = [
            ['remix', '-'],
            ['lowpass', f'{self.sr//2}'],
            ['rate', f'{self.sr}'],
        ]
        audio, _ = torchaudio.sox_effects.apply_effects_file(
            audio_path,
            effects=effects,
            channels_first=True
        )
        audio = torchaudio.functional.preemphasis(audio, coeff=0.97)
        return audio

    def __getitem__(self, idx: int):
        data_id = self.index_map[idx]
        datum = self.data_dict[data_id]
        audio = self._load_audio(datum['wav'])
        if self.target_size and audio.size(-1) > self.target_size:
            audio = _truncate(audio, self.target_size)
        return data_id, audio.unsqueeze(0)


class SpectrogramDataset(BaseAudioDataset):
    def __init__(
        self,
        data: str,
        config: Dict,
    ):
        super().__init__(data, config)

        self.spec = Spectrogram(
                n_fft=config['n_fft'],
                win_length=config['win_len'],
                hop_length=config['hop_len'],
                center=False
        )

        self.transforms = MelScale(
            config['n_mels'],
            config['sample_rate'],
            n_stft=config['n_fft']//2+1
        ) if config['n_mels'] else None
 
        self.simclr = True if config['model'] == 'simclr' else False

    def __getitem__(self, idx: int):
        data_id = self.index_map[idx]
        datum = self.data_dict[data_id]
        audio = self._load_audio(datum['wav'])
        if self.target_size and audio.size(-1) > self.target_size:
            audio = _truncate(audio, self.target_size)
        spec = self.spec(audio)
        if self.transforms:
            spec = self.transforms(spec)
        if self.simclr:
            # For SimCLR we need two augmentations of a same sample
            spec = spec.expand(2, -1, -1)
        return data_id, spec


def _pad_batch(batch: List[Tensor]) -> Tuple[Tensor, List[int]]:
    lengths = [x.size(-1) for x in batch]
    max_len = max(lengths)
    batch = [_apply_padding(x, max_len) for x in batch]
    return torch.cat(batch, dim=0), lengths


def padded_batch_collate(
    batch: List[Tuple[str, Tensor]]
) -> Tuple[Tuple[str], Tensor, List[int]]:
    lbl, batch = zip(*batch)
    batch, lengths = _pad_batch(batch)
    return lbl, batch, lengths


def simclr_batch_collate(
    batch: List[Tuple[str, Tensor]],
) -> Tuple[List[str], Tensor, Tensor, List[int]]:
    lbl, batch = zip(*batch)
    batch, lengths = _pad_batch(batch)
    evens_mask = torch.arange(len(batch)) % 2 == 0
    odds_mask = torch.arange(len(batch)) % 2 != 0
    return lbl, batch[evens_mask], batch[odds_mask], lengths


if __name__ == '__main__':
    config = {
        'data_file': '../atthack/atthack_test.json',
        'sample_rate': 16000,
        'n_fft': 512,
        'win_len': 512,
        'hop_len': 256,
        'simclr': True,
        'target_size': 16000,
        'n_mels': None
    }
    spec_aug = SpecAugment(
            n_time_masks=1,
            time_mask_param=16,
            n_freq_masks=1,
            freq_mask_param=64,
            iid_masks=True,
            p=1.0,
            zero_masking=True
        )
    # dataset = BaseAudioDataset(config)
    data = SpectrogramDataset(config)
    loader = DataLoader(
        data,
        3,
        collate_fn=simclr_batch_collate
    )

    # for i in range(len(dataset)):
    #     label, data = dataset[i]

    for Y, X1, X2, L in loader:
        X1, X2 = spec_aug(X1), spec_aug(X2)
        print(Y, L, X1.mean(), X2.mean(), X1.size())


# Unit tests
import pytest

@pytest.fixture
def simclr_dataset():
    return SpectrogramDataset({
        'data_file': '../atthack/atthack_test.json',
        'sample_rate': 16000,
        'n_fft': 512,
        'win_len': 512,
        'hop_len': 256,
        'simclr': True,
        'target_size': 16000,
        'n_mels': None
    })

def test_simclr_dataset_getitem(simclr_dataset):
    _, x = simclr_dataset[0]
    assert len(x.size()) == 3 and x.size(0) == 2 and x.size(1) == 512//2+1
    assert torch.equal(x[0], x[1])

def test_simclr_batch_collate(simclr_dataset):
    batch = [simclr_dataset[i] for i in range(10)]
    simclr_batch = simclr_batch_collate(batch)
    assert torch.equal(simclr_batch[1], simclr_batch[2])

@pytest.fixture
def dataset():
    return SpectrogramDataset({
        'data_file': '../atthack/atthack_test.json',
        'sample_rate': 16000,
        'n_fft': 512,
        'win_len': 512,
        'hop_len': 256,
        'simclr': False,
        'target_size': 16000,
        'n_mels': None
    })

@pytest.fixture
def mel_dataset():
    return SpectrogramDataset({
        'data_file': '../atthack/atthack_test.json',
        'sample_rate': 16000,
        'n_fft': 512,
        'win_len': 512,
        'hop_len': 256,
        'simclr': False,
        'target_size': 16000,
        'n_mels': 80


    })

def test_dataset_getitem(dataset):
    _, x = dataset[0]
    assert len(x.size()) == 3 and x.size(0) == 1 and x.size(1) == 512//2+1

def test_mel_dataset_getitem(mel_dataset):
    _, x = mel_dataset[0]
    assert len(x.size()) == 3 and x.size(0) == 1 and x.size(1) == 80

def test_padded_batch_collate(dataset):
    batch = [dataset[i] for i in range(10)]
    max_len = max([x.size(-1) for _, x in batch])
    padded_batch = padded_batch_collate(batch)
    assert all([x.size(-1) == max_len for x in padded_batch[1]])

