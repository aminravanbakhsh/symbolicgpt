# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
),

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

print(parent_dir)
print(parent_parent_dir)

from dimension_analyst import Dimensional_Analyst
from formula import Formula
from unit import Unit

# load libraries
import glob
import json
import math
import pickle
import random
import numpy as np

#from tqdm import tqdm
from numpy import * # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from scipy.optimize import minimize, least_squares

from helpers import set_seed, sample_from_model
from helpers import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

# set the random seed
set_seed(42)