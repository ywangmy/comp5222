#
# Created on Fri Nov 24 2023 00:21:43
# Author: Mukai (Tom Notch) Yu
# Email: myual@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
