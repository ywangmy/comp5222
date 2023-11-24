#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --model='wrgat' --match_threshold=0.1 --fraction=0.01 --gnn_layers=4 --show_keypoints --viz --fast_viz --batch_size=4 --learning_rate=1e-4 --max_keypoints=48
# 9l 1b 32k = 4588
# 64k = 17Mb
# 4l 2b 32k = 12910
# 4l 4b 32k = 25212
# 4l 4b 48k 0.1 0.5 0.5 0.5 = 32930
