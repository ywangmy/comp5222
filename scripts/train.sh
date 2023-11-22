#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --model='gat' --match_threshold=0.1 --fraction=0.001 --gnn_layers=3 --show_keypoints --viz --fast_viz --batch_size=4
