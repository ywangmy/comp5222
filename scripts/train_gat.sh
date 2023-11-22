#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --model='gat' --match_threshold=0.1 --fraction=0.05 --gnn_layers=5 --show_keypoints --viz --fast_viz --batch_size=32
