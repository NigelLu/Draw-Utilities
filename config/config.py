"""module config.py
"""
# * log related
SHOULD_LOG = False

# * denormalize
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]

# * pt file related
PT_FILE_PATH = '/scratch/xl3139/Draw-Utilities/pt-files/drawList-0-temp10-200.pt'
SAVE_DIR = '/scratch/xl3139/Draw-Utilities/output/task-definition'

# * draw config
SPLIT = 0
START_IDX = 0
NUM_TO_DRAW = 100

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4
