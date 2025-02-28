# coding: utf-8

from collections import namedtuple

ModelConfig = namedtuple(
    'ModelConfig',
    field_names=[
        'batch_size', 'learning_rate', 'weight_decay', 'epochs',
        'warmup_steps', 'temperature',
        'input_dim', 'n_heads', 'ffn_dim', 'n_layers', 'kernel_size',
        'steps', 'noise_samples'
    ],
    rename=False,
    defaults=[8, 1e-5, 1e-5, 1, 5, 0.1, 80, 4, 128, 2, 31, 12, 6]
)

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
