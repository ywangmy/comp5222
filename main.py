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
from torch.autograd import Variable
from tqdm import tqdm

import wandb
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
    parser.add_argument("--mode", type=str, choices={"train", "test"})
    parser.add_argument("--model_config_name", type=str)
    parser.add_argument("--load_epoch", type=int, default=None)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )

    opt = parser.parse_args()
    return opt


def get_model_str(config):
    superglue_config = config["superglue"]
    return f"{superglue_config['model_name']}-({config['train']['dataset']['COCO']['fraction']}|{config['train']['learning_rate']}-{superglue_config['batch_size']})-{superglue_config['match_threshold']}-{superglue_config['max_keypoints']}-{superglue_config['num_gnn_layers']}x{superglue_config['graph_type']}-{superglue_config['edge_pool']==None}"


def get_model_ckpt_path(config, epoch):
    return Path("ckpt") / f"{get_model_str(config)}/model_epoch_{epoch}.pth"


def get_superglue_config(config, epoch=None):
    superglue_config = config["superglue"]
    superglue_config.update(
        {
            "load_ckpt_path": None
            if epoch == None
            else get_model_ckpt_path(config, epoch),
            "GNN_layers": (
                ["self", "cross"]
                if config["superglue"]["graph_type"] == 2
                else ["union"]
            )
            * config["superglue"]["num_gnn_layers"],
        }
    )
    return superglue_config


# def train(config, superglue_config, get_model_str_func):
def train(config):
    torch.set_grad_enabled(True)
    torch.multiprocessing.set_sharing_strategy("file_system")

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

    superglue = SuperGlue(config["superglue"])

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

            # Visualization
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
                    input["image0"].cpu().numpy() * 255.0,
                    input["image1"].cpu().numpy() * 255.0,
                )

                kpts0, kpts1 = (
                    input["keypoints0"].cpu().numpy(),
                    input["keypoints1"].cpu().numpy(),
                )
                matches, conf = (
                    output["matches0"][0].cpu().detach().numpy(),
                    output["matching_scores0"][0].cpu().detach().numpy(),
                )
                image0 = read_image_modified(
                    image0,
                    config["train"]["dataset"]["COCO"]["resize"],
                    config["resize_float"],
                )
                image1 = read_image_modified(
                    image1,
                    config["train"]["dataset"]["COCO"]["resize"],
                    config["resize_float"],
                )
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


def test(config):
    # torch.set_grad_enabled(True)
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # store viz results
    output_dir = Path(config["test"]["output_dir"]) / get_model_str(config)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))

    # torch.autograd.set_detect_anomaly(True)

    # load training data
    feature_extractor = FeatureExtractor(config["feature_extraction"])
    perspective_warper = PerspectiveWarper(config["perspective_warper"])

    train_loader = FeatureMatchingDataLoader(
        config["test"]["dataset"], feature_extractor, perspective_warper
    )

    superglue = SuperGlue(config["superglue"])
    superglue.eval()
    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")

    num_iters = len(train_loader)
    prec_list = []
    rec_list = []
    print("num iters", num_iters)
    for i, input in enumerate(pbar := tqdm(train_loader, total=num_iters)):
        for k in input:
            if k != "file_name" and k != "image0" and k != "image1":
                if type(input[k]) == torch.Tensor:
                    input[k] = Variable(input[k].cuda())
                else:
                    input[k] = Variable(torch.stack(input[k]).cuda())

        output = superglue(input)  # originally not .float()
        batch_size = input["partial_assignment_matrix"].shape[0]
        for b in range(batch_size):
            prec, rec = compute_precision_recall(
                input["partial_assignment_matrix"][b], output["matches0"][b]
            )
            prec_list.append(prec)
            rec_list.append(rec)
        pbar.set_description(
            f"p {torch.stack(prec_list).mean()} r {torch.stack(rec_list).mean()}"
        )
    prec_mean = torch.stack(prec_list).mean()
    rec_mean = torch.stack(rec_list).mean()
    print("Final precison:", prec_mean.item())
    print("Final recall:", rec_mean.item())


def construct_confusion_matrix(input_partial_assignment_matrix, output_matches0):
    input_assignment_matrix = input_partial_assignment_matrix[:-1, :-1]
    output_assignment_matrix = torch.zeros_like(input_assignment_matrix)
    for i in range(len(output_matches0)):
        if not output_matches0[i] == -1:
            output_assignment_matrix[i, output_matches0[i]] = 1

    mutual_assignment_matrix = torch.logical_and(
        input_assignment_matrix, output_assignment_matrix
    )

    TP = torch.sum(mutual_assignment_matrix)

    FP = torch.sum(output_assignment_matrix) - TP

    FN = torch.sum(input_assignment_matrix) - TP

    confusion_matrix = torch.zeros((2, 2))
    confusion_matrix[0, 0] = TP
    confusion_matrix[0, 1] = FP
    confusion_matrix[1, 0] = FN

    return confusion_matrix


def compute_precision_recall(input_partial_assignment_matrix, output_matches0):
    confusion_matrix = construct_confusion_matrix(
        input_partial_assignment_matrix, output_matches0
    )

    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = torch.tensor(0)
    else:
        recall = TP / (TP + FN)

    return precision, recall


def aggregate_configs(opt, config, config_model):
    import warnings

    # config_model = config['superglue']
    config_model["model_config_name"] = opt.model_config_name
    # config_model = config_model[config_model['model_config_name']]

    config["mode"] = opt.mode
    config["eval"] = opt.eval

    config["superglue"].update(config_model)  # Overwrite
    config["superglue"] = get_superglue_config(config, epoch=opt.load_epoch)

    # Unify ...
    # - batch_size
    config["train"]["dataset"]["batch_size"] = config["superglue"]["batch_size"]
    # config["test"]["dataset"]["batch_size"] = config["superglue"]["batch_size"]
    # - descriptor_dim
    for key in ["descriptor_dim", "max_keypoints"]:
        if config["feature_extraction"][key] != config["superglue"][key]:
            warnings.warn(f"Unifying {key}")
            config["feature_extraction"][key] = config["superglue"][key]

    return config


def main():
    # Configuration
    opt = get_args()

    config = read_file(opt.config)  # General configs
    config_model = read_file(
        f"./configs/superglue/{opt.model_config_name}.yaml"
    )  # Model-specific configs
    # config_model = config["superglue"]["config"]
    config = aggregate_configs(opt, config, config_model)
    print("Processed config:")
    print(config)
    print(config["superglue"])

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

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "test":
        test(config)


if __name__ == "__main__":
    main()
