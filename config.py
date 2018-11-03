import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Date: Aug 20th, 2018
# Description: contains network configuration parameters
# Input: None
# Output: None
#----------------------------------------------------------------------------------------------------------------------
#yolo parameters
#size -> 448 x 448 -> (64x64) x 7x7
DATA_PATH = 'training'
NETWORK_PATH = 'network'
CACHE_PATH = os.path.join(NETWORK_PATH, 'cache')
OUTPUT_DIR = os.path.join(NETWORK_PATH, 'output')
WEIGHTS_DIR = os.path.join(NETWORK_PATH, 'weights')
LABEL_PATH = 'Anemone_canadensis_label.txt'
WEIGHTS_FILE = None
CLASSES = ['bud', 'flower', 'fruit', 'Others', 'Others', 'Others',
           'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others'] # we will only use the first three class

# model parameter
#image size
IMAGE_SIZE = 448
CELL_NUM = 7
# WARNING: we now assume that every cell has only one object
CENTERS_PER_CELL = 2
ALPHA = 0.1
DISP_CONSOLE = False
OBJECT_SCALE = 2.0
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
MAX_ITER = 30000
SUMMARY_ITER = 100
SAVE_ITER = 10000
#
# test parameter
# 
THRESHOLD = 0.5   #0.33 better than random selection
DIST_THRESHOLD = 0.5 #0.5
