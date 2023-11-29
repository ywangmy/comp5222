#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python main.py --mode='test' --epoch=100 --input_dir yfcc/raw_data/yfcc100m --output_dir dump_yfcc_test_results --nms_radius 3 --resize_float\
#     --model='ori' --match_threshold=0.1 --fraction=0.01 --num_gnn_layers=9 --show_keypoints --viz --fast_viz --batch_size=32
CUDA_VISIBLE_DEVICES=1 python main.py --mode='test' --eval --model='ori' --epoch=100 --match_threshold=0.1 --fraction=0.01 --num_gnn_layers=9 --show_keypoints --viz --fast_viz --batch_size=32 --input_dir assets/scannet_sample_images --input_pairs assets/scannet_sample_pairs_with_gt.txt --output_dir dump_scannet_test_results --eval
#
