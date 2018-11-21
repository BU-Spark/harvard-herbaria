import sys
import cv2 as cv
import tensorflow as tf
from alexNet import AlexNet
import numpy as np
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for detecting feature on a single image
# Input: a filename
# Output: a .jpg file with feature labeled
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
weights_file = "model/model_epoch401.ckpt"
# validation patch location
IMAGENET_MEAN = [123.68, 116.779, 103.939]
# Learning params
learning_rate = 0.001
num_epochs = 500
batch_size = 225
# Network params
dropout_rate = 0.5
num_classes = 4
train_layers = ['fc8', 'fc7', 'fc6']
patch_size = 227
def main(argv):
    # load and preprocess the image
    origin_img = cv.imread(argv[0])
    if origin_img is None:
        print("image does not exist")
        return 0
    #mean subtraction
    img = np.subtract(origin_img, IMAGENET_MEAN)
    #tile the image
    tiled_image = np.ones((batch_size, patch_size, patch_size, 3), dtype=np.float32)
    # partition images into 225 patches
    patch_width = int(np.floor(img.shape[1] / 15))
    patch_height = int(np.floor(img.shape[0] / 15))
    for row in range(0, 15):
        for col in range(0, 15):
            patch = img[row * patch_height:(row + 1) * patch_height, col * patch_width:(col + 1) * patch_width]
            patch = cv.resize(patch, (patch_size, patch_size))
            tiled_image[row * 15 + col] = patch

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, train_layers)

    # Link variable to model output
    score = model.fc8

    with tf.name_scope("predict"):
        predict = score

    # Start Tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + weights_file)
        saver = tf.train.Saver()
        saver.restore(sess, weights_file)
        prediction = sess.run(predict, feed_dict={x: tiled_image, keep_prob: 1.})
    # find the label with highest prediction value
    pred_label = np.argmax(prediction, axis=1)
    for row in range(0, 15):
        for col in range(0, 15):
            pred = pred_label[row * 15 + col]
            color = (0, 0, 255) if pred == 0 else ((0, 255, 0) if pred == 1 else (255, 0, 0))
            # only draw none background
            if pred != 3:
                cv.rectangle(origin_img, (col * patch_width, row * patch_height), ((col + 1) * patch_width, (row + 1) * patch_height), color, 3)
    cv.imwrite("output.jpg", origin_img)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])