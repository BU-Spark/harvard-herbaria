import sys
import random
import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for splitting the label into training and testing
# Input: a filename
# Output: 2 files: training labels and testing labels
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
# percentage of data used for training
training_percent = 0.9
def main(argv):
    training_labels = open("patch_labels/training_labels.txt", "w")
    testing_labels = open("patch_labels/testing_labels.txt", "w")
    labels = []
    with open("patch_labels/patch_labels.txt", "r") as patch_labels:
        line = patch_labels.readline()
        while line:
            labels.append(line)
            line = patch_labels.readline()
        # shuffle the lines of labels
        random.shuffle(labels)
        training_num = int(len(labels) * training_percent)
        testing_num = len(labels) - training_num
        for line in labels[0:training_num]:
            training_labels.write(line)
        for line in labels[training_num : training_num + testing_num]:
            testing_labels.write(line)
    training_labels.close()
    testing_labels.close()
    patch_labels.close()
if __name__ == "__main__":
    main(sys.argv)