#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --mode='train' --model_config_name='rgat-4-1'
# 9l 1b 32k = 4588
# 64k = 17Mb
# 4l 2b 32k = 12910
# 4l 4b 32k = 25212
# 4l 4b 48k 0.1 0.5 0.5 0.5 = 32930
