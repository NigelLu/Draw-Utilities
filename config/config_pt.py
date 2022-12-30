""" module config.config_pt.py
"""
import torch
import random

import numpy as np


# * set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# * log related
SHOULD_LOG = False

# * denormalize
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]

# * draw config
SPLIT = 0
NUM_TO_DRAW = 200
DRAW_LABEL = False

# * pt file related
PT_FILE_PATH = f'pt-files/draw-info-pascal-1210_f0_pm1{SPLIT}-200.pt'
SAVE_DIR = 'output/mask-pt'

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4