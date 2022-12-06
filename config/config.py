"""module config.py
"""
# * log related
SHOULD_LOG = False

# * denormalize
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]

# * draw config
SPLIT = 3

# * pt file related
PT_FILE_PATH = f'/scratch/xl3139/Draw-Utilities/pt-files/drawList-{SPLIT}-temp10-200.pt'
SAVE_DIR = '/scratch/xl3139/Draw-Utilities/output/task-definition'

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4