# coding: utf-8

from typing import Optional
from collections import namedtuple
from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    epochs: int = 1

    batch_size: int = 8
    lr: float = 1e-5
    weight_decay: float = 1e-5

    temperature: float = 0.1

    input_dim: int = 80
    n_heads: int = 4
    ffn_dim: int = 128
    n_layers: int = 2
    kernel_size: int = 31

    steps: int = 8
    noise_samples: int = 12


@dataclass
class AudioConfiguration:
    sample_rate: int = 16000
    n_fft: int = 512
    win_len: int = 512
    hop_len: int = 256

    n_mels: Optional[int] = 80

    n_freq_mask: int = 2
    max_freq_mask: int = 16
    n_time_mask: int = 2
    max_time_mask: int = 64

    target_size: int = 16000*5


