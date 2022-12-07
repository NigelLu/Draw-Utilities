"""module config.config_pt.py
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

# * pt file related
PT_FILE_PATH = f'pt-files/drawList-{SPLIT}-temp10-200.pt'
SAVE_DIR = 'output/task-definition-pt'

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4