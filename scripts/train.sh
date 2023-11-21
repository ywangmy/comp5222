#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --show_keypoints --data_fraction=0.001 --eval_output_dir='dump_match_pairs0001'
