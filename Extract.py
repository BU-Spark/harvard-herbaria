import sys
import random
import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for splitting the label into different categories
# Input: a filename
# Output: 14 files: containing the patch name and labels
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
# percentage of data used for training
label_names = ['Anemone_canadensis_label.txt', 'Anemone_hepatica_label.txt', 'Aquilegia_canadensis_label.txt',
               'Bidens_vulgata_label.txt', 'Celastrus_orbiculatus_label.txt', 'Centaurea_stoebe_label.txt',
               'Cirsium_arvense_label.txt', 'Cirsium_discolor_label.txt', 'Geranium_maculatum_label.txt',
               'Geranium_robertianum_label.txt', 'Hemerocallis_fulva_label.txt', 'Hibiscus_moscheutos_label.txt',
               'Impatiens_capensis_label.txt', 'Iris_pseudacorus_label.txt']
category_names = ['Anemone_canadensis', 'Anemone_hepatica', 'Aquilegia_canadensis',
               'Bidens_vulgata', 'Celastrus_orbiculatus', 'Centaurea_stoebe',
               'Cirsium_arvense', 'Cirsium_discolor', 'Geranium_maculatum',
               'Geranium_robertianum', 'Hemerocallis_fulva', 'Hibiscus_moscheutos',
               'Impatiens_capensis', 'Iris_pseudacorus']
training_percent = 0.9
def main(argv):
    # create the file array
    filewritor = []
    for category in label_names:
        writor = open("patch_labels/" + category, "w")
        filewritor.append(writor)
    labels = []
    # read from training
    with open("patch_labels/training_labels.txt", "r") as patch_labels:
        line = patch_labels.readline()
        while line:
            labels.append(line)
            line = patch_labels.readline()
    # read from testing
    with open("patch_labels/testing_labels.txt", "r") as patch_labels:
        line = patch_labels.readline()
        while line:
            labels.append(line)
            line = patch_labels.readline()
    # shuffle the lines of labels
    random.shuffle(labels)
    for patch in labels:
        name = patch.split(".")[0]
        index = category_names.index(name)
        filewritor[index].write(patch)
    # close all file writors
    for writor in filewritor:
        writor.close()
    patch_labels.close()
if __name__ == "__main__":
    main(sys.argv)