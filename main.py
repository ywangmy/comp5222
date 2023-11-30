#!/usr/bin/env python3
#
# Created on Wed Nov 29 2023 23:31:36
# Author: Mukai (Tom Notch) Yu, Yicheng Wang
# Email: myual@connect.ust.hk, ywangmy@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yicheng Wang
#
import argparse
from pathlib import Path

import matplotlib.cm as cm
import torch.multiprocessing
import wandb
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.feature_extractor import FeatureExtractor
from dataloader.general import FeatureMatchingDataLoader
from dataloader.perspective_warper import PerspectiveWarper
from match_pairs import test
from models.superglue import SuperGlue
from models.utils import make_matching_plot
from models.utils import read_image_modified
from utils.files import read_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, default="./configs/default.yaml")
    parser.add_argument("--load_epoch", type=int, default=None)

    opt = parser.parse_args()
    return opt


def get_model_str(config):
    superglue_config = config["Superglue"]
    return f"{superglue_config['model_name']}-({config['train']['dataset']['COCO']['fraction']}|{config['train']['learning_rate']}-{config['train']['dataset']['batch_size']})-{superglue_config['match_threshold']}-{superglue_config['max_keypoints']}-{superglue_config[superglue_config['model_name']]['num_gnn_layers']}x{superglue_config[superglue_config['model_name']]['graph_type']}-{superglue_config[superglue_config['model_name']]['edge_pool']==None}"


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


# def train(config, superglue_config, get_model_str_func):
def train(config):
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

    # if (
    #     config["model"]["model_name"] == "rgat"
    #     or config["model"]["model_name"] == "wrgat"
    # ):
    #     config["model"]["graph_type"] = 1
    # config['model']['graph_type']
    # config = {
    #     # "superpoint": config['feature_extraction']['Superpoint'],
    #     "superglue": superglue_config,
    # }

    # store viz results
    output_dir = Path(config["train"]["output_dir"]) / get_model_str(config)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))

    # torch.autograd.set_detect_anomaly(True)

    # load training data
    feature_extractor = FeatureExtractor(config["feature_extraction"])
    perspective_warper = PerspectiveWarper(config["perspective_warper"])

    train_loader = FeatureMatchingDataLoader(
        config["train"]["dataset"], feature_extractor, perspective_warper
    )

    superglue = SuperGlue(config["Superglue"])

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

    config = read_file(opt.config)  # General configs
    # config_model = read_file(
    #     f"./configs/{opt.model_config_name}.yaml"
    # )  # Model-specific configs
    # config_model = config["model"]["config"]
    # config = aggregate_configs(opt, config, config_model)
    # print("Processed config:")
    # print(config)

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "test":
        # test(
        #     opt,
        #     get_superglue_config(config, config["model"]["load_epoch"]),
        #     get_model_ckpt_path(opt, config["model"]["load_epoch"]),
        # )
        test(config)


if __name__ == "__main__":
    main()
