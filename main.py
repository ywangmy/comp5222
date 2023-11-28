import argparse
from pathlib import Path

import matplotlib.cm as cm
import torch.multiprocessing
from torch.autograd import Variable
from tqdm import tqdm

import wandb
from load_data import SparseDataset
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from models.utils import make_matching_plot
from models.utils import read_image_modified


def get_args():
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", choices={"train", "test"}, default="train")

    parser.add_argument(
        "--viz", action="store_true", help="Visualize the matches and dump the plots"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )

    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="indoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=48,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )

    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )

    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )
    parser.add_argument(
        "--resize_float",
        action="store_true",
        help="Resize the image after casting uint8 to float",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Skip the pair if output .npz files are already found",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Plot the keypoints in addition to the matches",
    )
    parser.add_argument(
        "--fast_viz",
        action="store_true",
        help="Use faster image visualization based on OpenCV instead of Matplotlib",
    )
    parser.add_argument(
        "--viz_extension",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Visualization file extension. Use pdf for highest-quality.",
    )

    parser.add_argument(
        "--opencv_display",
        action="store_true",
        help="Visualize via OpenCV before saving output images",
    )
    parser.add_argument(
        "--input_pairs",
        type=str,
        default="assets/scannet_sample_pairs_with_gt.txt",
        help="Path to the list of image pairs for evaluation",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ordering of pairs before processing",
    )
    parser.add_argument(
        "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/scannet_sample_images/",
        help="Path to the directory that contains the images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dump_match_pairs/",
        help="Path to the directory in which the .npz results and optional,"
        "visualizations are written",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument(
        "--train_path",
        type=str,
        default="./COCO2014/train2014/",
        help="Path to the directory of training imgs.",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epoches")

    parser.add_argument("--descriptor_dim", type=int, default=128)

    parser.add_argument("--fraction", type=float, default=1.0)

    parser.add_argument("--model", default="gat")
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--graph", type=int, default=2)
    parser.add_argument("--edge_pool", type=list, default=None)
    parser.add_argument("--epoch", type=int, default=None)

    opt = parser.parse_args()
    return opt


def get_model_str(opt):
    return f"{opt.model}-({opt.fraction}|{opt.learning_rate}-{opt.batch_size})-{opt.match_threshold}-{opt.max_keypoints}-{opt.gnn_layers}x{opt.graph}-{opt.edge_pool==None}"


def get_model_ckpt_path(opt, epoch):
    return Path("ckpt") / f"{get_model_str(opt)}/model_epoch_{epoch}.pth"


def get_superglue_config(opt, epoch=None):
    return {
        "model": opt.model,
        "load_ckpt": None if epoch == None else get_model_ckpt_path(opt, epoch),
        "sinkhorn_iterations": opt.sinkhorn_iterations,
        "match_threshold": opt.match_threshold,
        "descriptor_dim": 128,
        "weights": "indoor",
        "keypoint_encoder": [32, 64, 128],
        "GNN_layers": (["self", "cross"] if opt.graph == 2 else ["union"])
        * opt.gnn_layers,
        "sinkhorn_iterations": 100,
    }


def main():
    opt = get_args()
    if opt.mode == "train":
        from train import train

        train(opt, get_superglue_config(opt), get_model_str)
    elif opt.mode == "test":
        from match_pairs import test

        test(
            opt,
            get_superglue_config(opt, opt.epoch),
            get_model_ckpt_path(opt, opt.epoch),
        )


if __name__ == "__main__":
    main()
