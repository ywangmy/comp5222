#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import wandb
from load_data import SparseDataset
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from models.utils import AverageTimer
from models.utils import compute_epipolar_error
from models.utils import compute_pose_error
from models.utils import error_colormap
from models.utils import estimate_pose
from models.utils import make_matching_plot
from models.utils import pose_auc
from models.utils import read_image
from models.utils import read_image_modified
from models.utils import rotate_intrinsics
from models.utils import rotate_pose_inplane
from models.utils import scale_intrinsics
from torch.autograd import Variable
from tqdm import tqdm

# from models.matchingForTraining import MatchingForTraining


def train(opt, superglue_config, get_model_str_func):
    torch.set_grad_enabled(True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # make sure the flags are properly used
    assert not (
        opt.opencv_display and not opt.viz
    ), "Must use --viz with --opencv_display"
    assert not (
        opt.opencv_display and not opt.fast_viz
    ), "Cannot use --opencv_display without --fast_viz"
    assert not (opt.fast_viz and not opt.viz), "Must use --viz with --fast_viz"
    assert not (
        opt.fast_viz and opt.viz_extension == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    if opt.model == "rgat" or opt.model == "wrgat":
        opt.graph = 1
    opt.graph
    config = {
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
            "descriptor_dim": 256,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": -1,
            "remove_borders": 4,
        },
        "superglue": superglue_config,
    }

    # store viz results
    output_dir = Path(opt.output_dir) / get_model_str_func(opt)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))

    # torch.autograd.set_detect_anomaly(True)

    # load training data
    train_set = SparseDataset(
        opt.train_path, opt.max_keypoints, opt.fraction, opt.resize
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        shuffle=False,
        batch_size=opt.batch_size,
        drop_last=True,
        # collate_fn=train_set.collate_fn,
    )

    superglue = SuperGlue(config.get("superglue", {}))

    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="comp5222",
        # track hyperparameters and run metadata
        config={
            "max_keypoints": opt.max_keypoints,
            "resize": opt.resize,
            "learning_rate": opt.learning_rate,
            "fraction": opt.fraction,
            "batch_size": opt.batch_size,
        }
        | config["superglue"],
    )

    # start training

    for epoch in range(1, opt.num_epochs + 1):
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
                        opt.num_epochs,
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
                image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = output_dir / "{}_matches.{}".format(
                    str(i), opt.viz_extension
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
                    opt.show_keypoints,
                    opt.fast_viz,
                    opt.opencv_display,
                    "Matches",
                )

            # process checkpoint for every 5e3 images
            if (i + 1) % 5e3 == 0:
                model_out_path = "model_epoch_{}.pth".format(epoch)
                torch.save(superglue, model_out_path)
                print(
                    "Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}".format(
                        epoch, opt.num_epochs, i + 1, len(train_loader), model_out_path
                    )
                )

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        model_out_path = get_model_ckpt_path(opt, epoch)
        model_out_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(superglue, model_out_path)
        print(
            "Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}".format(
                epoch, opt.num_epochs, epoch_loss, model_out_path
            )
        )


if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    train(opt)
