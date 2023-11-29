#
# Created on Fri Nov 24 2023 00:21:43
# Author: Mukai (Tom Notch) Yu
# Email: myual@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from .COCO import COCODataset
from .ScanNet import ScanNetDataset


class FeatureMatchingDataLoader(DataLoader):
    def __init__(self, config, feature_extractor, perspective_warper, drop_last=True):
        self.config = config
        self.batch_size = int(config["batch_size"])
        self.shuffle = bool(config["shuffle"])

        self.feature_extractor = feature_extractor
        self.perspective_warper = perspective_warper

        dataset_list = []
        if "COCO" in config:
            dataset_list.append(
                COCODataset(
                    config["COCO"], self.feature_extractor, self.perspective_warper
                )
            )
        if "ScanNet" in config:
            dataset_list.append(
                ScanNetDataset(
                    config["ScanNet"], self.feature_extractor, self.perspective_warper
                )
            )

        super().__init__(
            ConcatDataset(dataset_list),
            self.batch_size,
            shuffle=self.shuffle,
            drop_last=drop_last,
        )
