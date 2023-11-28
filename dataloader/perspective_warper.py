#
# Created on Mon Nov 27 2023 20:39:10
# Author: Mukai (Tom Notch) Yu
# Email: myual@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
from abc import ABC
from abc import abstractclassmethod

import cv2
import numpy as np
import torch


class PerspectiveWarper(ABC):
    def __new__(cls, config):
        if config is None:
            raise ValueError("Missing perspective warper config")

        if "homography" in config:
            return super(PerspectiveWarper, cls).__new__(HomographyWarper)
        elif "nerf" in config:
            return super(PerspectiveWarper, cls).__new__(NerFWarper)
        else:
            raise ValueError(
                "Unsupported perspective warper type "
                + next(iter(config))
                + " in config"
            )

    def __init__(self, config):
        self.max_warp_match_pixel_distance = int(
            config["max_warp_match_pixel_distance"]
        )

    @abstractclassmethod
    def generate_transform(self, width, height):
        pass

    @abstractclassmethod
    def warp_image_to_novel(self, image, warp_transform):
        pass

    @abstractclassmethod
    def warp_keypoints_to_novel(self, keypoints, warp_transform):
        pass

    @abstractclassmethod
    def warp_keypoints_to_original(self, keypoints, warp_transform):
        pass


class HomographyWarper(PerspectiveWarper):
    def __init__(self, config):
        super().__init__(config)
        self.perturbation_threshold = float(
            config["homography"]["perturbation_threshold"]
        )
        self.random_rotation = bool(config["homography"]["random_rotation"])

    def generate_transform(self, width, height):
        corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

        # Generate random perturbations for x (width) and y (height) separately
        warp_x = np.random.randint(
            -width * self.perturbation_threshold,
            width * self.perturbation_threshold,
            size=(4, 1),
        ).astype(np.float32)
        warp_y = np.random.randint(
            -height * self.perturbation_threshold,
            height * self.perturbation_threshold,
            size=(4, 1),
        ).astype(np.float32)
        warp = np.hstack((warp_x, warp_y))  # Combine x and y perturbations

        new_corners = corners + warp
        transform = cv2.getPerspectiveTransform(corners, new_corners)

        # Randomly decide whether to apply rotation
        if self.random_rotation:
            # Generate a random angle between -180 and 180 degrees
            angle = np.random.uniform(-180, 180)
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Convert rotation matrix to 3x3 homography matrix
            rotation_transform = np.vstack([rotation_matrix, [0, 0, 1]])

            # Combine homography and rotation transformations
            transform = rotation_transform @ transform

        return [transform, (width, height)]

    def warp_image_to_novel(self, image, warp_transform):
        return cv2.warpPerspective(image, warp_transform[0], warp_transform[1])

    def warp_keypoints_to_novel(self, keypoints, warp_transform):
        # Ensure keypoints is a tensor and warp_transform is a tuple containing the homography matrix and size
        homography_matrix, _ = warp_transform
        homography_matrix = torch.from_numpy(homography_matrix).float()

        # Convert keypoints to homogeneous coordinates by adding a dimension of ones
        num_keypoints = keypoints.shape[0]
        ones = torch.ones(
            num_keypoints, 1, device=keypoints.device, dtype=keypoints.dtype
        )
        homogeneous_keypoints = torch.cat([keypoints, ones], dim=-1)

        # Apply the homography matrix to the keypoints
        # We transpose the homography matrix to align with the dimension of homogeneous_keypoints for matrix multiplication
        warped_homogeneous_keypoints = homogeneous_keypoints @ homography_matrix.T

        # Convert back from homogeneous to Cartesian coordinates
        warped_keypoints = warped_homogeneous_keypoints[
            :, :2
        ] / warped_homogeneous_keypoints[:, 2].unsqueeze(1)

        return warped_keypoints

    def warp_keypoints_to_original(self, keypoints, warp_transform):
        # Get the homography matrix and its inverse
        homography_matrix, _ = warp_transform
        homography_matrix = torch.from_numpy(homography_matrix).float()
        homography_inverse = torch.inverse(homography_matrix)

        # Convert keypoints to homogeneous coordinates by adding a dimension of ones
        num_keypoints = keypoints.shape[0]
        ones = torch.ones(
            num_keypoints, 1, device=keypoints.device, dtype=keypoints.dtype
        )
        homogeneous_keypoints = torch.cat([keypoints, ones], dim=-1)

        # Apply the inverse homography matrix to the keypoints
        original_homogeneous_keypoints = homogeneous_keypoints @ homography_inverse.T

        # Convert back from homogeneous to Cartesian coordinates
        original_keypoints = original_homogeneous_keypoints[
            :, :2
        ] / original_homogeneous_keypoints[:, 2].unsqueeze(1)

        return original_keypoints


class NerFWarper(PerspectiveWarper):
    def __init__(self, config):
        super().__init__(config)
        self.novel_vew_max_angle = float(config["nerf"]["novel_view_max_angle"])

    def generate_transform(self, width, height):
        raise NotImplementedError

    def warp_image_to_novel(self, image, warp_transform):
        raise NotImplementedError

    def warp_keypoints_to_novel(self, keypoints, warp_transform):
        raise NotImplementedError

    def warp_keypoints_to_original(self, keypoints, warp_transform):
        raise NotImplementedError
