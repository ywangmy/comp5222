import datetime
import math
import os

import cv2
import numpy as np
import torch
from models.superpoint import SuperPoint
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


def resize(img, resize):
    img_h, img_w = img.shape[0], img.shape[1]
    cur_size = max(img_h, img_w)
    if len(resize) == 1:
        scale1, scale2 = resize[0] / cur_size, resize[0] / cur_size
    else:
        scale1, scale2 = resize[0] / img_h, resize[1] / img_w
    new_h, new_w = int(img_h * scale1), int(img_w * scale2)
    new_img = cv2.resize(img.astype("float32"), (new_w, new_h)).astype("uint8")
    scale = np.asarray([scale2, scale1])
    return new_img, scale


class ExtractSIFT:
    def __init__(self, nfeatures, padding: bool = False):
        self.nfeatures = nfeatures
        self.padding = padding
        self.sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=self.nfeatures  # , contrastThreshold=0.00000
        )

    def run(self, image):
        sift = self.sift

        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        # kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        # kp1_num = min(self.nfeatures, len(kp1))

        kp1_num = self.nfeatures
        kp1 = kp1[:kp1_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])

        if len(kp1_np) < 64:
            # print(len(kp1_np))
            if self.padding:
                res = int(self.nfeatures - len(kp1_np))
                pad_kp = (
                    np.random.uniform(size=[res, 2])
                    * (image.shape[0] + image.shape[1])
                    / 2
                )
                pad_scroes1 = np.zeros([res])  # scores := 0
                pad_desc1 = np.zeros((res, 128))

                if len(kp1_np) == 0:
                    kp1_np = pad_kp
                    scores1_np = pad_scroes1
                    descs1 = pad_desc1
                else:
                    kp1_np = np.concatenate([kp1_np, pad_kp], axis=0)
                    scores1_np = np.concatenate([scores1_np, pad_scroes1], axis=0)
                    descs1 = np.concatenate([descs1, pad_desc1], axis=0)
                # print(kp1_np)
                # print(descs1)
                # print(scores1_np)
        kp1_np = kp1_np[:kp1_num, :]
        descs1 = descs1[:kp1_num, :]
        return kp1_np, descs1, scores1_np


class ExtractSuperpoint(object):
    def __init__(self, config):
        default_config = {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "detection_threshold": config["det_th"],
            "max_keypoints": config["num_kpt"],
            "remove_borders": 4,
            "model_path": "../weights/sp/superpoint_v1.pth",
        }
        self.superpoint_extractor = SuperPoint(default_config)
        self.superpoint_extractor.eval(), self.superpoint_extractor.cuda()
        self.num_kp = config["num_kpt"]
        if "padding" in config.keys():
            self.padding = config["padding"]
        else:
            self.padding = False
        self.resize = config["resize"]

    def run(self, img):
        # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        scale = 1
        if self.resize[0] != -1:
            img, scale = resize(img, self.resize)
        with torch.no_grad():
            result = self.superpoint_extractor(
                torch.from_numpy(img / 255.0).float()[None, None].cuda()
            )
        score, kpt, desc = (
            result["scores"][0],
            result["keypoints"][0],
            result["descriptors"][0],
        )
        score, kpt, desc = score.cpu().numpy(), kpt.cpu().numpy(), desc.cpu().numpy().T
        kpt = np.concatenate([kpt / scale, score[:, np.newaxis]], axis=-1)
        # padding randomly
        if self.padding:
            if len(kpt) < self.num_kp:
                res = int(self.num_kp - len(kpt))  # number of remaining
                pad_kpt = (
                    np.random.uniform(size=[res, 2]) * (img.shape[0] + img.shape[1]) / 2
                )
                pad_desc = np.random.uniform(size=[res, 256])
                pad_desc = pad_desc / np.linalg.norm(pad_desc, axis=-1)[:, np.newaxis]

                score = np.concatenate([score, np.zeros(res)], axis=0)
                # pad_kpt = np.concatenate([pad_x, np.zeros([res, 1])], axis=-1) # scores := 0

                kpt = np.concatenate([kpt, pad_kpt], axis=0)
                desc = np.concatenate([desc, pad_desc], axis=0)
        return kpt, desc, score


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures, fraction=0.1, resize=None):
        all_files = [train_path + f for f in os.listdir(train_path)]

        # Select a fraction of the total files
        subset_size = int(len(all_files) * fraction)
        self.files = all_files[:subset_size]
        self.nfeatures = nfeatures
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
        self.resize = resize

        self.extractor = ExtractSIFT(nfeatures, True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if self.resize != None:
            image = cv2.resize(
                image.astype("float32"), (self.resize[0], self.resize[1])
            ).astype("uint8")

        width, height = image.shape[:2]
        corners = np.array(
            [[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32
        )
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(
            src=image, M=M, dsize=(image.shape[1], image.shape[0])
        )  # return an image type

        #

        kp1_np, descs1, scores1_np = self.extractor.run(image)
        kp2_np, descs2, scores2_np = self.extractor.run(warped)
        # print(kp1_np.shape, descs1.shape, scores1_np.shape)
        # print(kp2_np.shape, descs2.shape, scores2_np.shape)
        # skip this image pair if no keypoints detected in image
        if (
            len(kp1_np) <= 1 or len(kp2_np) <= 1
        ):  # https://github.com/yingxin-jia/SuperGlue-pytorch/issues/31, originally < <
            return {
                "keypoints0": torch.zeros([0, 0, 2], dtype=torch.float),
                "keypoints1": torch.zeros([0, 0, 2], dtype=torch.float),
                "descriptors0": torch.zeros([0, 2], dtype=torch.float),
                "descriptors1": torch.zeros([0, 2], dtype=torch.float),
                "image0": image,
                "image1": warped,
                "file_name": file_name,
            }

        # obtain the matching matrix of the image pair
        # matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate(
            [
                missing1[np.newaxis, :],
                (len(kp2_np)) * np.ones((1, len(missing1)), dtype=np.int64),
            ]
        )
        MN3 = np.concatenate(
            [
                (len(kp1_np)) * np.ones((1, len(missing2)), dtype=np.int64),
                missing2[np.newaxis, :],
            ]
        )
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        pad_matches = -1 * np.ones(
            (2, self.nfeatures * 2 - all_matches.shape[1]), dtype=np.int64
        )
        all_matches = np.concatenate([all_matches, pad_matches], axis=1)

        # kp1_np = kp1_np.reshape((1, -1, 2))
        # kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.0)
        descs2 = np.transpose(descs2 / 256.0)

        image = torch.from_numpy(image / 255.0).float()[None].cuda()
        warped = torch.from_numpy(warped / 255.0).float()[None].cuda()
        # print(image.shape)
        return {
            "keypoints0": torch.from_numpy(kp1_np),
            "keypoints1": torch.from_numpy(kp2_np),
            "descriptors0": torch.from_numpy(descs1),
            "descriptors1": torch.from_numpy(descs2),
            "scores0": torch.from_numpy(scores1_np),
            "scores1": torch.from_numpy(scores2_np),
            "image0": image,
            "image1": warped,
            "image0_shape": image.shape,
            "image1_shape": warped.shape,
            "all_matches": torch.from_numpy(all_matches),
            "file_name": file_name,
        }

    def collate_fn(self, batch):
        # print("# Enter collate_fn")
        # print(len(batch))
        # for k, v in batch[0].items():
        #     if k != 'file_name':
        #         print(k, v.shape)
        # batch format:
        #       0:
        #           'keypoints0': [a numpy array with shape (nkeypoints, 2)]
        #           'keypoints1': [a numpy array with shape (nkeypoints, 2)]
        #           'descriptors0': [a numpy array with shape (64)], length = nkeypoints
        #           'descriptors1': [a numpy array with shape (64)], length = nkeypoints
        #           'scores0': [a 1 element numpy array with shape ()] length = nkeypoints
        #           'scores1': [a 1 element numpy array with shape ()] length = nkeypoints
        #           'image0': [a numpy array with shape (1, height, width)]

        # Initialize lists to hold the batch data
        keypoints0, keypoints1 = [], []
        descriptors0, descriptors1 = [], []
        scores0, scores1 = [], []
        images0, images1 = [], []
        all_matches = []
        file_names = []

        # Go through each sample and append the data to the lists
        for item in batch:
            keypoints0.append(item["keypoints0"])
            keypoints1.append(item["keypoints1"])
            descriptors0.append(item["descriptors0"])
            descriptors1.append(item["descriptors1"])
            scores0.append(item["scores0"])
            scores1.append(item["scores1"])
            images0.append(item["image0_shape"])
            images1.append(item["image1_shape"])
            all_matches.append(item["all_matches"])
            # file_names.append(item["file_name"])

        # Convert lists to tensors or stack them appropriately
        keypoints0 = torch.stack(keypoints0, dim=0)
        keypoints1 = torch.stack(keypoints1, dim=0)
        descriptors0 = torch.stack(descriptors0, dim=0)
        descriptors1 = torch.stack(descriptors1, dim=0)
        scores0 = torch.stack(scores0, dim=0)
        scores1 = torch.stack(scores1, dim=0)
        images0 = torch.stack(images0, dim=0)
        images1 = torch.stack(images1, dim=0)
        all_matches = torch.stack(all_matches, dim=0)

        # Return a dictionary with the batched data
        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "descriptors0": descriptors0,
            "descriptors1": descriptors1,
            "scores0": scores0,
            "scores1": scores1,
            "image0_shape": images0,
            "image1_shape": images1,
            "all_matches": all_matches,
            # "file_names": file_names,
        }
