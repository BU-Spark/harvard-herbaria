import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import config as cfg
from YOLO_network import YOLONet
from timer import Timer
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Date: Aug 20th, 2018
# Description: This is the file for testing the object detector
# Input: Single image in RGB
# Output: Images with points marked
#----------------------------------------------------------------------------------------------------------------------

class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_num = cfg.CELL_NUM
        self.centers_per_cell = cfg.CENTERS_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.dist_threshold = cfg.DIST_THRESHOLD
        self.boundary1 = self.cell_num * self.cell_num * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_num * self.cell_num * self.centers_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

# ----------------------------------------------------------------------------------------------------------------------
# Visualize the detected features on the input image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            # determine the object type
            color_code = (0,0,0)
            if (result[i][0] == 0):
                # bud -> red
                color_code = (0, 0, 255)
            elif (result[i][0] == 1):
                # flower -> green
                color_code = (0, 255, 0)
            else:
                # fruit -> blue
                color_code = (255, 0, 0)
            cv2.circle(img, (x, y), 63, color_code, 3)

# ----------------------------------------------------------------------------------------------------------------------
# try to detect features on the given input image, one image
# ----------------------------------------------------------------------------------------------------------------------
    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        # scaling back to the the size of input image
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)

        return result

# ----------------------------------------------------------------------------------------------------------------------
# Pass the image through the network
# ----------------------------------------------------------------------------------------------------------------------
    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            result = self.interpret_output(net_output[i])
            results.append(result)
        return results

# ----------------------------------------------------------------------------------------------------------------------
# Process the prediction by applying non maxima suppression
# ----------------------------------------------------------------------------------------------------------------------
    def interpret_output(self, output):
        # predicted conditional probabilities
        probs = np.zeros((self.cell_num, self.cell_num,
                          self.centers_per_cell, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_num, self.cell_num, self.num_class))
        # predicted object probability
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_num, self.cell_num, self.centers_per_cell))
        # predicted coordinate
        centers = np.reshape(
            output[self.boundary2:],
            (self.cell_num, self.cell_num, self.centers_per_cell, 2))
        offset = np.array(
            [np.arange(self.cell_num)] * self.cell_num * self.centers_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.centers_per_cell, self.cell_num, self.cell_num]),
            (1, 2, 0))
        # scale the predictions back to the original input image size
        centers[:, :, :, 0] += offset
        centers[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        centers[:, :, :, :2] = 1.0 * centers[:, :, :, 0:2] / self.cell_num

        centers *= self.image_size

        for i in range(self.centers_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        #compute the unconditional class probability and throw the predictions with low probility.
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_centers = np.nonzero(filter_mat_probs)
        centers_filtered = centers[filter_mat_centers[0],
                               filter_mat_centers[1], filter_mat_centers[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_centers[0], filter_mat_centers[1], filter_mat_centers[2]]

        # sort the probability from high to low
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        centers_filtered = centers_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(centers_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(centers_filtered)):
                if self.calc_dist(centers_filtered[i], centers_filtered[j]) < self.dist_threshold:
                    probs_filtered[j] = 0.0

        # create the filtering mask
        filter_dist = np.array(probs_filtered > 0.0, dtype='bool')
        centers_filtered = centers_filtered[filter_dist]
        probs_filtered = probs_filtered[filter_dist]
        classes_num_filtered = classes_num_filtered[filter_dist]

        result = []
        # format the output
        for i in range(len(centers_filtered)):
            result.append(
                [classes_num_filtered[i],      # integer class type
                 centers_filtered[i][0],
                 centers_filtered[i][1],
                 probs_filtered[i]])

        return result

# ----------------------------------------------------------------------------------------------------------------------
# Compute the distance of two centers
# Center1 and center2 are 2-d vector
# ----------------------------------------------------------------------------------------------------------------------
    def calc_dist(self, center1, center2):
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        cell_length = self.image_size/self.cell_num
        distance = distance / (2 * cell_length)
        distance = np.clip(distance, 0.0, 1.0)
        # clip the value
        return 1 - distance

# ----------------------------------------------------------------------------------------------------------------------
# detect the objects in the image
# ----------------------------------------------------------------------------------------------------------------------
    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        #save the output image
        cv2.imwrite("output.jpg", image)

# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='2018_08_29_11_06/yolo.ckpt-1', type=str)
    parser.add_argument('--weight_dir', default='output', type=str)
    parser.add_argument('--data_dir', default="network", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # construct the network and load trained weights
    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir,args.weights)
    detector = Detector(yolo, weight_file)

    # detect from image file
    imname = 'training/Anemone_canadensis.88485.3642.jpg'
    detector.image_detector(imname)

if __name__ == '__main__':
    main()
