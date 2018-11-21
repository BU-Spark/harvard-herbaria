import sys
import cv2 as cv
import random
import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for extracting mini patches from labeled images
#              For each image, we generate 4 mini-patches: flower, fruit, bud and background.
# Input: label files
# Output: mini-patches
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
patch_size = 227
def main(argv):
    label_names = ['Anemone_canadensis_label.txt','Anemone_hepatica_label.txt', 'Aquilegia_canadensis_label.txt',
                   'Bidens_vulgata_label.txt', 'Celastrus_orbiculatus_label.txt', 'Centaurea_stoebe_label.txt',
                   'Cirsium_arvense_label.txt', 'Cirsium_discolor_label.txt', 'Geranium_maculatum_label.txt',
                   'Geranium_robertianum_label.txt', 'Hemerocallis_fulva_label.txt', 'Hibiscus_moscheutos_label.txt',
                   'Impatiens_capensis_label.txt', 'Iris_pseudacorus_label.txt']

    label_names = ['Anemone_canadensis_label.txt']
    if not os.path.exists("Anemone_canadensis"):
        os.makedirs("Anemone_canadensis")
    # .txt for storing the patch label
    patch_label = open("patch_labels/Anemone_canadensis_patch_labels.txt", "w")

    for label_name in label_names:
        specie = label_name.split("_")
        specie = specie[0] + "_" + specie[1]
        # create directory
        if not os.path.exists("Anemone_canadensis/" + specie):
            os.makedirs("Anemone_canadensis/" + specie)
        name = 'labels/' + label_name
        with open(name, "r") as original:
            line = original.readline()
            while line:
                line = line.split("\n")[0]
                line = line.split(",")
                buds = line[1].split(";")
                flowers = line[2].split(";")
                fruits = line[3].split(";")
                # get image name
                imname = "images/" + specie + "/" + line[0]
                # detect from image file
                img = cv.imread(imname)
                specie = imname.split("/")[1]
                img_height, img_width, channels = img.shape
                # randomly pick one if not empty
                #bud
                if buds[0] != ' ':
                    for item in buds:
                        index = buds.index(item)
                        center = item.split("_")
                        center[0] = int(float(center[0]))
                        center[1] = int(float(center[1]))
                        # the coordinate has to be within the
                        topleft_x = int(max((center[0] - patch_size/2), 0))
                        topleft_y = int(max((center[1] - patch_size/2), 0))
                        bottomright_x = int(min((center[0] + patch_size/2), img_width))
                        bottomright_y = int(min((center[1] + patch_size/2), img_height))
                        patch = img[topleft_y:bottomright_y, topleft_x:bottomright_x]
                        # write the label to file
                        patch_file = imname[:-4] + "." + str(index) +".0.jpeg"
                        patch_file = patch_file.split("/")[-1]
                        patch_label.write(patch_file + " " + "0" + "\n")
                        # save patch
                        patch_file = "Anemone_canadensis/" + specie + "/" + patch_file
                        cv.imwrite(patch_file, patch)
                # randomly pick one if not empty
                if flowers[0] != ' ':
                    for item in flowers:
                        index = flowers.index(item)
                        center = item.split("_")
                        center[0] = int(float(center[0]))
                        center[1] = int(float(center[1]))
                        # the coordinate has to be within the
                        topleft_x = int(max(((center[0]) - patch_size / 2), 0))
                        topleft_y = int(max(((center[1]) - patch_size / 2), 0))
                        bottomright_x = int(min(((center[0]) + patch_size / 2), img_width))
                        bottomright_y = int(min(((center[1]) + patch_size / 2), img_height))
                        patch = img[topleft_y:bottomright_y, topleft_x:bottomright_x]
                        # write the label to file
                        patch_file = imname[:-4] + "." + str(index) + ".1.jpeg"
                        patch_file = patch_file.split("/")[-1]
                        patch_label.write(patch_file + " " + "1" + "\n")
                        # save patch
                        patch_file = "Anemone_canadensis/" + specie + "/" + patch_file
                        cv.imwrite(patch_file, patch)
                if fruits[0] != ' ':
                    for item in fruits:
                        index = fruits.index(item)
                        center = item.split("_")
                        center[0] = int(float(center[0]))
                        center[1] = int(float(center[1]))
                        # the coordinate has to be within the
                        topleft_x = int(max((center[0] - patch_size / 2), 0))
                        topleft_y = int(max((center[1] - patch_size / 2), 0))
                        bottomright_x = int(min((center[0] + patch_size / 2), img_width))
                        bottomright_y = int(min((center[1] + patch_size / 2), img_height))
                        patch = img[topleft_y:bottomright_y, topleft_x:bottomright_x]
                        # write the label to file
                        patch_file = imname[:-4] + "." + str(index) + ".2.jpeg"
                        patch_file = patch_file.split("/")[-1]
                        patch_label.write(patch_file + " " + "2" + "\n")
                        # save patch
                        patch_file = "Anemone_canadensis/" + specie + "/" + patch_file
                        cv.imwrite(patch_file, patch)
                # randomly pick a background patch
                rand_x = random.randint(0, img_width)
                rand_y = random.randint(0, img_height)
                topleft_x = int(max((rand_x - patch_size / 2), 0))
                topleft_y = int(max((rand_y - patch_size / 2), 0))
                bottomright_x = int(min((rand_x + patch_size / 2), img_width))
                bottomright_y = int(min((rand_y + patch_size / 2), img_height))
                patch = img[topleft_y:bottomright_y, topleft_x:bottomright_x]
                # write the label to file
                imname = imname[:-4] + ".3.jpeg"
                patch_file = imname.split("/")[-1]
                patch_label.write(patch_file + " " + "3" + "\n")
                # save patch
                patch_file = "Anemone_canadensis/" + specie + "/" + patch_file
                cv.imwrite(patch_file, patch)
                line = original.readline()
            original.close()
    patch_label.close()
if __name__ == "__main__":
    main(sys.argv)