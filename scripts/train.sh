#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --model='gat' --match_threshold=0.1 --fraction=0.0003 --gnn_layers=3 --show_keypoints --viz --fast_viz
