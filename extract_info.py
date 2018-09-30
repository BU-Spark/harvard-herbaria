import sys
import csv
from collections import deque
import urllib.request
import os
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Date: July 28th, 2018
# Description: This is the executable for
#        1. extract unique URLs i.e. unique images
#        2. extract unique genus and specie name
#        3. extract the trustworthy images (not implemented yet)
#        4. download the images and place them into correct directories
# Input: None
# Output: txt files containing the urls, labels, coordinates of the image. line by line
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
FILENAME = "thoreaus-annotations.csv"
#Iris pseudacorus
#Hibiscus moscheutos
#Aquilegia canadensis
def main(argv):
    # create image directory
    if not os.path.exists("images"):
        os.makedirs("images")
    # file to save
    urls = deque()
    names = deque()
    count = 0
    #open csv
    with open(FILENAME, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #skip the first line
            if count > 0:
                url = row[2]
                #parse the name of the specimen
                image_name = url.split("/")[-1]
                specie_name = image_name.split(".")[0]
                # store unique specie names
                if specie_name not in names:
                    names.append(specie_name)
                    # create sub directories for each species
                    if not os.path.exists("images/" + specie_name):
                        os.makedirs("images/" + specie_name)
                #store unique urls
                if url not in urls:
                    urls.append(url)
                    #download this image
                    path = "images/" + specie_name + "/" + image_name
                    urllib.request.urlretrieve(url, path)
            else:
                count += 1
    print(len(urls))
    return 0
if __name__ == "__main__":
    main(sys.argv)