"""module config.py
"""
# * log related
SHOULD_LOG = False

# * denormalize
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]

# * draw config
SPLIT = 0

# * pt file related
PT_FILE_PATH = f'pt-files/drawList-{SPLIT}-temp10-200.pt'
SAVE_DIR = 'output/task-definition'

GAMMA_SCALAR = 1
MASK_WEIGHT = 0.4