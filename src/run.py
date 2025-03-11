#!/usr/bin/env Python3
# coding: utf-8

from typing import Dict
import argparse
from dataclasses import asdict

from training import train
from config import AudioConfiguration, ModelConfiguration

def main(args: Dict):
    model_config = ModelConfiguration()
    audio_config = AudioConfiguration()
    config = asdict(model_config) | asdict(audio_config) | args
    train(args['train_data'], args['val_data'], config)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('train_data', help='path to train json file')
    arg_parser.add_argument('val_data', help='path to validation json file')

    arg_parser.add_argument('--model', default='simclr', choices=['simclr', 'cpc'])
    arg_parser.add_argument('--epochs', default=1, type=int)
    arg_parser.add_argument('--lr', default=1e-5, type=float)

    arg_parser.add_argument('--input-dim', default=80, type=int)
    arg_parser.add_argument('--n-fft', default=512, type=int)
    arg_parser.add_argument('--n-mels', default=80, type=int)

    args = vars(arg_parser.parse_args())
    main(args)
