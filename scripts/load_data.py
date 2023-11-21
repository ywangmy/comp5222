import datetime
import math
import os

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures, fraction=0.1):
        all_files = [train_path + f for f in os.listdir(train_path)]

        # Select a fraction of the total files
        subset_size = int(len(all_files) * fraction)
        self.files = all_files[:subset_size]

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        sift = self.sift
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

        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        # kp1_num = kp2_num = self.nfeatures
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        if (
            len(kp1) <= 1 or len(kp2) <= 1
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

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])
        scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        # obtain the matching matrix of the image pair
        matched = self.matcher.match(descs1, descs2)
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
                (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64),
            ]
        )
        MN3 = np.concatenate(
            [
                (len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64),
                missing2[np.newaxis, :],
            ]
        )
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        # kp1_np = kp1_np.reshape((1, -1, 2))
        # kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.0)
        descs2 = np.transpose(descs2 / 256.0)

        image = torch.from_numpy(image / 255.0).float()[None].cuda()
        warped = torch.from_numpy(warped / 255.0).float()[None].cuda()

        return {
            "keypoints0": torch.from_numpy(kp1_np),
            "keypoints1": torch.from_numpy(kp2_np),
            "descriptors0": torch.from_numpy(descs1),
            "descriptors1": torch.from_numpy(descs2),
            "scores0": torch.from_numpy(scores1_np),
            "scores1": torch.from_numpy(scores2_np),
            "image0": image,
            "image1": warped,
            "all_matches": torch.from_numpy(all_matches),
            "file_name": file_name,
        }

    def collate_fn(self, batch):
        # batch format:
        #       0:
        #           'keypoints0': [a numpy array with shape (nkeypoints, 2)]
        #           'keypoints1': [a numpy array with shape (nkeypoints, 2)]
        #           'descriptors0': [a numpy array with shape (64)], length = nkeypoints
        #           'descriptors1': [a numpy array with shape (64)], length = nkeypoints
        #           'scores0': [a 1 element numpy array with shape ()] length = nkeypoints
        #           'scores1': [a 1 element numpy array with shape ()] length = nkeypoints
        #           'image0': [a numpy array with shape (1, height, width)]
        import pdb

        pdb.set_trace()

        # Initialize lists to hold the batch data
        keypoints0, keypoints1 = [], []
        descriptors0, descriptors1 = [], []
        scores0, scores1 = [], []
        images0, images1 = [], []
        all_matches = []
        file_names = []

        # Go through each sample and append the data to the lists
        for item in batch:
            keypoints0.append(torch.FloatTensor(item["keypoints0"]))
            keypoints1.append(torch.FloatTensor(item["keypoints1"]))
            descriptors0.append(torch.FloatTensor(item["descriptors0"]))
            descriptors1.append(torch.FloatTensor(item["descriptors1"]))
            scores0.append(torch.FloatTensor(item["scores0"]))
            scores1.append(torch.FloatTensor(item["scores1"]))
            images0.append(item["image0"])
            images1.append(item["image1"])
            all_matches.append(torch.LongTensor(item["all_matches"]))
            file_names.append(item["file_name"])

        # Convert lists to tensors or stack them appropriately
        keypoints0 = torch.cat(keypoints0, dim=0)
        keypoints1 = torch.cat(keypoints1, dim=0)
        descriptors0 = torch.cat(descriptors0, dim=0)
        descriptors1 = torch.cat(descriptors1, dim=0)
        scores0 = torch.cat(scores0, dim=0)
        scores1 = torch.cat(scores1, dim=0)
        images0 = torch.stack(images0, dim=0)
        images1 = torch.stack(images1, dim=0)
        all_matches = torch.cat(all_matches, dim=0)

        # Return a dictionary with the batched data
        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "descriptors0": descriptors0,
            "descriptors1": descriptors1,
            "scores0": scores0,
            "scores1": scores1,
            "image0": images0,
            "image1": images1,
            "all_matches": all_matches,
            "file_names": file_names,
        }
