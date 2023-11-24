#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --model='ori' --match_threshold=0.1 --fraction=0.01 --gnn_layers=9 --show_keypoints --viz --fast_viz --batch_size=32 --learning_rate=1e-4
#
