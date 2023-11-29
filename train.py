#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch.multiprocessing
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

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

# from models.matchingForTraining import MatchingForTraining
