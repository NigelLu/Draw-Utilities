""" module config.config_pascal.py
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

# * draw config
SPLIT = 0
NUM_TO_DRAW = 100
DRAW_LABEL = True

# * pt file related
PT_FILE_PATH = f'pt-files/drawList-{SPLIT}-temp10-200.pt'
SAVE_DIR = 'output/mask-raw'

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4