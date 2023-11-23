#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py --model='wrgat' --match_threshold=0.01 --fraction=0.1 --gnn_layers=9 --show_keypoints --viz --fast_viz --batch_size=2 --learning_rate=1e-3 --max_keypoints=32
# 9l 1b 32k = 4588
# 64k = 17Mb
