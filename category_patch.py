import sys
import random
import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for extracting training and testing labels, category by category
# Input: a filename
# Output: 2 files: training labels and testing labels
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
# percentage of data used for training
training_percent = 0.8
def main(argv):
    label_names = ['Anemone_canadensis_label.txt','Anemone_hepatica_label.txt', 'Aquilegia_canadensis_label.txt',
                   'Bidens_vulgata_label.txt', 'Celastrus_orbiculatus_label.txt', 'Centaurea_stoebe_label.txt',
                   'Cirsium_arvense_label.txt', 'Cirsium_discolor_label.txt', 'Geranium_maculatum_label.txt',
                   'Geranium_robertianum_label.txt', 'Hemerocallis_fulva_label.txt', 'Hibiscus_moscheutos_label.txt',
                   'Impatiens_capensis_label.txt', 'Iris_pseudacorus_label.txt']
    for name in label_names:
        category_name = name.split(".")[0]
        training_labels = open("patch_labels/category_patch_label/" + category_name + "_training" + ".txt", "w")
        testing_labels = open("patch_labels/category_patch_label/" + category_name + "_testing" + ".txt", "w")
        labels = []
        with open("patch_labels/" + name, "r") as patch_labels:
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
            for line in labels[training_num: training_num + testing_num]:
                testing_labels.write(line)
        training_labels.close()
        testing_labels.close()
        patch_labels.close()
if __name__ == "__main__":
    main(sys.argv)