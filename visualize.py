import sys
import csv
import numpy as np
import cv2
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Date: Aug 19th, 2018
# Description: This is the executable for visualising points on the image
# Input: None
# Output: images with buds, fruits and flowers circled
# Assumptions:
#               1.Entries for coordinates of objects are ordered as bud, flower, fruit. Coordinates from the same labels
#                 are splitted by semicolon and the coordinates of different labels are splitted by comma. And the x,y
#                 coordinates are spiltted by underscore
#               2. Bud, flower and fruit points are drawn as red, green and blue accordingly
#               3.executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
FILENAME = "training.txt"
def main(argv):
    #find the minimal dimension across the dataset
    min_x = 10000
    min_y = 10000
    # read the coordinates
    with open(FILENAME, "r") as ground_truth:
        line = ground_truth.readline()
        while line:
            line = line.split(",")
            image_path = "training/" + line[0] + ".jpg"
            #input image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            #find image size
            current_y, current_x = img.shape[0:2]
            if (current_x < min_x):
                min_x = current_x
            if (current_y < min_y):
                min_y = current_y
            #coordinate lists
            bud = line[1].split(";")

            flower = line[2].split(";")
            fruit = line[3].split("\n")[0].split(";")
            #draw bud
            for item in bud:
                #only tries to read the coordinates when they are not empty
                if item != " ":
                    x = int(item.split("_")[0])
                    y = int(item.split("_")[1])
                    cv2.circle(img, (x, y), 63, (0, 0, 255), 3)
            #draw flower
            for item in flower:
                if item != " ":
                    x = int(item.split("_")[0])
                    y = int(item.split("_")[1])
                    cv2.circle(img, (x, y), 63, (0, 255, 0), 3)
            #draw flower
            for item in fruit:
                if item != " ":
                    x = int(item.split("_")[0])
                    y = int(item.split("_")[1])
                    cv2.circle(img, (x, y), 63, (255, 0, 0), 3)
            #save image
            cv2.imwrite("ground_truth/" + line[0] + ".jpg", img)
            #sample_size += 1
            line = ground_truth.readline()
    print(min_x,min_y)
if __name__ == "__main__":
    main(sys.argv)