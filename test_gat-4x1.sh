#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --mode='test' --model_config_name='gat-4-1' --load_epoch=100
#
