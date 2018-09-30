import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Date: Aug 20th, 2018
# Description: contains network configuration parameters
# Input: None
# Output: None
#----------------------------------------------------------------------------------------------------------------------
#yolo parameters
#1200 1666 -> 1152 x 1152 -> 576 x 576 -> (64x64)x9x9
DATA_PATH = 'training'
NETWORK_PATH = 'network'
CACHE_PATH = os.path.join(NETWORK_PATH, 'cache')
OUTPUT_DIR = os.path.join(NETWORK_PATH, 'output')
WEIGHTS_DIR = os.path.join(NETWORK_PATH, 'weights')
LABEL_PATH = 'Anemone_canadensis_label.txt'
WEIGHTS_FILE = None
CLASSES = ['bud', 'flower', 'fruit']

# model parameter
#image size
IMAGE_SIZE = 576
CELL_NUM = 9
# WARNING: we now assume that every cell has only one object
CENTERS_PER_CELL = 1
ALPHA = 0.1
DISP_CONSOLE = False
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0
#
# training parameter
#
GPU = ''
LEARNING_RATE = 0.0001
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True
#size of training samples batch
BATCH_SIZE = 16
MAX_ITER = 2500
SUMMARY_ITER = 100
SAVE_ITER = 500
#
# test parameter
# 
THRESHOLD = 0.2    #0.2
DIST_THRESHOLD = 0.5 #0.5
