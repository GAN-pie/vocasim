#!/usr/bin/env Python3
# coding: utf-8

from typing import Dict
import argparse

from training import train, ModelConfig
from data_utils import AudioConfig

def main(args: Dict):
    config = ModelConfig()._asdict() | AudioConfig()._asdict() | args
    train(config)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('train_data', help='path to train json file')
    arg_parser.add_argument('val_data', help='path to validation json file')

    args = vars(arg_parser.parse_args())
    main(args)
