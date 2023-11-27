#
# Created on Mon Nov 27 2023 20:37:03
# Author: Mukai (Tom Notch) Yu
# Email: myual@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
import os
import sys
from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np
import torch

# Calculate the relative path to the project root
project_root = os.path.join(os.path.dirname(__file__), "..")
project_root = os.path.normpath(project_root)  # Normalize the path
# Add the project root to sys.path
sys.path.append(project_root)

from models.superpoint import SuperPoint


class FeatureExtractor(ABC):
    def __new__(cls, config):
        # Use __new__ to create an instance of the appropriate subclass
        extractor_config = config["extractor"]
        if extractor_config is None:
            raise ValueError("Missing extractor config")

        if "SIFT" in extractor_config:
            return super(FeatureExtractor, cls).__new__(SiftExtractor)
        elif "superpoint" in extractor_config:
            return super(FeatureExtractor, cls).__new__(SuperpointExtractor)
        else:
            raise ValueError(
                "Unsupported feature extractor type "
                + next(iter(extractor_config))
                + " in config"
            )

    def __init__(self, config):
        # This will only be called if a subclass instance is not created in __new__
        # Initialize shared configuration parameters
        self.config = config
        self.max_keypoints = int(config["max_keypoints"])
        self.descriptor_dim = int(config["descriptor_dim"])

    @abstractmethod
    def __call__(self, image):
        # This will be called when the instance is called like a function, e.g.
        # feature_extractor = FeatureExtractor(config["feature_extraction"])
        # features = feature_extractor(image)
        pass


class SiftExtractor(FeatureExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.contrast_threshold = float(
            config["extractor"]["SIFT"]["contrast_threshold"]
        )
        self.edge_threshold = float(config["extractor"]["SIFT"]["edge_threshold"])
        self.sigma = float(config["extractor"]["SIFT"]["sigma"])

        self.sift = cv2.SIFT_create(
            nfeatures=self.max_keypoints,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma,
        )

    def __call__(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        confidence_scores = torch.tensor([k.response for k in keypoints]).unsqueeze(-1)
        keypoints = torch.tensor([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        descriptors = torch.tensor(descriptors)
        return keypoints, descriptors, confidence_scores


class SuperpointExtractor(FeatureExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.superpoint_config = config["extractor"]["superpoint"]
        self.superpoint_config["max_keypoints"] = self.max_keypoints
        self.superpoint_config["descriptor_dim"] = self.descriptor_dim
        self.superpoint_config["nms_radius"] = int(self.superpoint_config["nms_radius"])

        self.superpoint = SuperPoint(self.superpoint_config).eval()
        self.superpoint.load_state_dict(
            torch.load(self.superpoint_config["model_weight_path"])
        )

    def __call__(self, image):
        # Convert the input image to a torch tensor
        input_tensor = torch.from_numpy(image / 255.0).float()[None, None]

        # Pass the image through the SuperPoint model
        result = self.superpoint({"image": input_tensor})
        return (
            result["keypoints"][0],
            result["descriptors"][0].T,
            result["scores"][0][:, np.newaxis],
        )
