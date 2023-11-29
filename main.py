import argparse
from pathlib import Path

import matplotlib.cm as cm
import torch.multiprocessing
from torch.autograd import Variable
from tqdm import tqdm

import wandb
from dataloader.feature_extractor import FeatureExtractor
from dataloader.general import FeatureMatchingDataLoader
from dataloader.perspective_warper import PerspectiveWarper
from dataloader.visualization import visualize_matches
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

    parser.add_argument("--model_config_name")
    parser.add_argument("--mode", choices={"train", "test"}, default="train")

    # parser.add_argument(
    #     "--viz", action="store_true", help="Visualize the matches and dump the plots"
    # )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )

    # parser.add_argument(
    #     "--superglue",
    #     choices={"indoor", "outdoor"},
    #     default="indoor",
    #     help="SuperGlue weights",
    # )
    # parser.add_argument(
    #     "--max_keypoints",
    #     type=int,
    #     default=48,
    #     help="Maximum number of keypoints detected by Superpoint"
    #     " ('-1' keeps all keypoints)",
    # )

    # parser.add_argument(
    #     "--keypoint_threshold",
    #     type=float,
    #     default=0.005,
    #     help="SuperPoint keypoint detector confidence threshold",
    # )
    # parser.add_argument(
    #     "--nms_radius",
    #     type=int,
    #     default=4,
    #     help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    # )
    # parser.add_argument(
    #     "--sinkhorn_iterations",
    #     type=int,
    #     default=20,
    #     help="Number of Sinkhorn iterations performed by SuperGlue",
    # )
    # parser.add_argument(
    #     "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    # )

    # parser.add_argument(
    #     "--resize",
    #     type=int,
    #     nargs="+",
    #     default=[640, 480],
    #     help="Resize the input image before running inference. If two numbers, "
    #     "resize to the exact dimensions, if one number, resize the max "
    #     "dimension, if -1, do not resize",
    # )
    # parser.add_argument(
    #     "--resize_float",
    #     action="store_true",
    #     help="Resize the image after casting uint8 to float",
    # )

    # parser.add_argument(
    #     "--cache",
    #     action="store_true",
    #     help="Skip the pair if output .npz files are already found",
    # )
    # parser.add_argument(
    #     "--show_keypoints",
    #     action="store_true",
    #     help="Plot the keypoints in addition to the matches",
    # )
    # parser.add_argument(
    #     "--fast_viz",
    #     action="store_true",
    #     help="Use faster image visualization based on OpenCV instead of Matplotlib",
    # )
    # parser.add_argument(
    #     "--viz_extension",
    #     type=str,
    #     default="png",
    #     choices=["png", "pdf"],
    #     help="Visualization file extension. Use pdf for highest-quality.",
    # )

    # parser.add_argument(
    #     "--opencv_display",
    #     action="store_true",
    #     help="Visualize via OpenCV before saving output images",
    # )
    # parser.add_argument(
    #     "--input_pairs",
    #     type=str,
    #     default="assets/scannet_sample_pairs_with_gt.txt",
    #     help="Path to the list of image pairs for evaluation",
    # )
    # parser.add_argument(
    #     "--shuffle",
    #     action="store_true",
    #     help="Shuffle ordering of pairs before processing",
    # )
    # parser.add_argument(
    #     "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
    # )

    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     default="assets/scannet_sample_images/",
    #     help="Path to the directory that contains the images",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="dump_match_pairs/",
    #     help="Path to the directory in which the .npz results and optional,"
    #     "visualizations are written",
    # )

    # parser.add_argument(
    #     "--learning_rate", type=float, default=0.0001, help="Learning rate"
    # )

    # parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    # parser.add_argument(
    #     "--train_path",
    #     type=str,
    #     default="./COCO2014/train2014/",
    #     help="Path to the directory of training imgs.",
    # )
    # parser.add_argument("--num_epochs", type=int, default=100, help="Number of epoches")

    # parser.add_argument("--descriptor_dim", type=int, default=128)

    # parser.add_argument("--fraction", type=float, default=1.0)

    # parser.add_argument("--model", default="gat")
    # parser.add_argument("--num_gnn_layers", type=int, default=3)
    # parser.add_argument("--graph_type", type=int, default=2)
    # parser.add_argument("--edge_pool", type=list, default=None)
    parser.add_argument("--load_epoch", type=int, default=None)

    opt = parser.parse_args()
    return opt


def get_model_str(config):
    return f"{config['model']['model_name']}-({config['train']['dataset']['COCO']['fraction']}|{config['train']['learning_rate']}-{config['model']['batch_size']})-{config['model']['match_threshold']}-{config['model']['max_keypoints']}-{config['model']['num_gnn_layers']}x{config['model']['graph_type']}-{config['model']['edge_pool']==None}"


def get_model_ckpt_path(config, epoch):
    return Path("ckpt") / f"{get_model_str(config)}/model_epoch_{epoch}.pth"


def get_superglue_config(config, epoch=None):
    superglue_config = config["model"]
    superglue_config.update(
        {
            "load_ckpt_path": None
            if epoch == None
            else get_model_ckpt_path(config, epoch),
            "GNN_layers": (
                ["self", "cross"] if config["model"]["graph_type"] == 2 else ["union"]
            )
            * config["model"]["num_gnn_layers"],
        }
    )
    return superglue_config


def train(config, superglue_config, get_model_str_func):
    torch.set_grad_enabled(True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # make sure the flags are properly used
    assert not (
        config["opencv_display"] and not config["viz"]
    ), "Must use --viz with --opencv_display"
    assert not (
        config["opencv_display"] and not config["fast_viz"]
    ), "Cannot use --opencv_display without --fast_viz"
    assert not (
        config["fast_viz"] and not config["viz"]
    ), "Must use --viz with --fast_viz"
    assert not (
        config["fast_viz"] and config["viz_extension"] == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    if (
        config["model"]["model_name"] == "rgat"
        or config["model"]["model_name"] == "wrgat"
    ):
        config["model"]["graph_type"] = 1
    # config['model']['graph_type']
    # config = {
    #     # "superpoint": config['feature_extraction']['Superpoint'],
    #     "superglue": superglue_config,
    # }

    # store viz results
    output_dir = Path(config["train"]["output_dir"]) / get_model_str_func(config)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))

    # torch.autograd.set_detect_anomaly(True)

    # load training data
    feature_extractor = FeatureExtractor(config["feature_extraction"])
    perspective_warper = PerspectiveWarper(config["perspective_warper"])

    train_loader = FeatureMatchingDataLoader(
        config["train"]["dataset"], feature_extractor, perspective_warper
    )
    # train_set = SparseDataset(
    #     opt.train_path, config['model']['max_keypoints'], config['train']['dataset']['COCO']['fraction'], opt.resize
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_set,
    #     shuffle=False,
    #     batch_size=config['model']['batch_size'],
    #     drop_last=True,
    #     # collate_fn=train_set.collate_fn,
    # )

    superglue = SuperGlue(superglue_config)

    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(
        superglue.parameters(), lr=config["train"]["learning_rate"]
    )
    mean_loss = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="comp5222",
        # track hyperparameters and run metadata
        config=config,
    )

    # start training

    for epoch in range(1, config["train"]["num_epochs"] + 1):
        epoch_loss = 0
        # originally double
        superglue.float().train()
        num_iters = len(train_loader)
        last_plot_id = 0
        for i, input in enumerate(pbar := tqdm(train_loader, total=num_iters)):
            for k in input:
                if k != "file_name" and k != "image0" and k != "image1":
                    if type(input[k]) == torch.Tensor:
                        input[k] = Variable(input[k].cuda())
                    else:
                        input[k] = Variable(torch.stack(input[k]).cuda())

            output = superglue(input)  # originally not .float()
            for k, v in input.items():
                input[k] = v[0]
            # input = {**input, **output}

            if output["skip_train"] == True:  # image has no keypoint
                continue

            # process loss
            Loss = output["loss"]

            # print('Loss', Loss)
            # exit()

            epoch_loss += Loss.item()

            mean_loss.append(Loss)
            pbar.set_description(f"running ave. loss {epoch_loss / (i+1)}")
            wandb.log({"loss": Loss.item()})

            superglue.zero_grad()
            Loss.backward()
            optimizer.step()

            # for every 50 images, print progress and visualize the matches
            if (i + 1 - last_plot_id) > (int)(num_iters * 0.1):
                last_plot_id = i + 1
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch,
                        config["train"]["num_epochs"],
                        i + 1,
                        len(train_loader),
                        torch.mean(torch.stack(mean_loss)).item(),
                    )
                )
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = (
                    input["image0"].cpu().numpy()[0] * 255.0,
                    input["image1"].cpu().numpy()[0] * 255.0,
                )

                # import pdb; pdb.set_trace()
                kpts0, kpts1 = (
                    input["keypoints0"].cpu().numpy(),
                    input["keypoints1"].cpu().numpy(),
                )
                matches, conf = (
                    output["matches0"][0].cpu().detach().numpy(),
                    output["matching_scores0"][0].cpu().detach().numpy(),
                )
                image0 = read_image_modified(image0, opt.resize, config["resize_float"])
                image1 = read_image_modified(image1, opt.resize, config["resize_float"])
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = output_dir / "{}_matches.{}".format(
                    str(i), config["viz_extension"]
                )
                color = cm.jet(mconf)
                stem = input["file_name"]
                text = []

                make_matching_plot(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    mkpts0,
                    mkpts1,
                    color,
                    text,
                    viz_path,
                    stem,
                    stem,
                    config["show_keypoints"],
                    config["fast_viz"],
                    config["opencv_display"],
                    "Matches",
                )

            # process checkpoint for every 5e3 images
            if (i + 1) % 5e3 == 0:
                model_out_path = "model_epoch_{}.pth".format(epoch)
                torch.save(superglue, model_out_path)
                print(
                    "Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}".format(
                        epoch,
                        config["train"]["num_epochs"],
                        i + 1,
                        len(train_loader),
                        model_out_path,
                    )
                )

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        model_out_path = get_model_ckpt_path(config, epoch)
        model_out_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(superglue, model_out_path)
        print(
            "Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}".format(
                epoch, config["train"]["num_epochs"], epoch_loss, model_out_path
            )
        )


def aggregate_configs(opt, config, config_model):
    import warnings

    config_model["load_epoch"] = opt.load_epoch
    config["mode"] = opt.mode
    config["eval"] = opt.eval

    config["model"].update(config_model)  # Overwrite

    # Unify ...
    # - batch_size
    config["train"]["dataset"]["batch_size"] = config["model"]["batch_size"]
    # - descriptor_dim
    for key in ["descriptor_dim", "max_keypoints"]:
        if config["feature_extraction"][key] != config["model"][key]:
            warnings.warn(f"Unifying {key}")
            config["feature_extraction"][key] = config["model"][key]

    return config


def main():
    # Configuration
    opt = get_args()
    from utils.files import read_file, print_dict

    config = read_file("./configs/default.yaml")  # General configs
    config_model = read_file(
        f"./configs/{opt.model_config_name}.yaml"
    )  # Model-specific configs
    config = aggregate_configs(opt, config, config_model)
    print("Processed config:")
    print(config)

    if config["mode"] == "train":
        train(config, get_superglue_config(config), get_model_str)
    elif config["mode"] == "test":
        from match_pairs import test

        test(
            opt,
            get_superglue_config(config, config["model"]["load_epoch"]),
            get_model_ckpt_path(opt, config["model"]["load_epoch"]),
        )


if __name__ == "__main__":
    main()
