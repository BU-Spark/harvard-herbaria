import sys
import cv2 as cv
import tensorflow as tf
from alexNet import AlexNet
import numpy as np
#----------------------------------------------------------------------------------------------------------------------
# Author: Siqi Zhang
# Description: This is the executable for evaluating accuracy on test patches
# Input: a filename
# Output: 1. overall accuracy  2. None background accuracy.
# Assumptions:
#               executable and csv are in the same directory
#----------------------------------------------------------------------------------------------------------------------
weights_file = "model/model_epoch301.ckpt"
# validation patch location
label_names = ['Anemone_canadensis_label.txt', 'Anemone_hepatica_label.txt', 'Aquilegia_canadensis_label.txt',
               'Bidens_vulgata_label.txt', 'Celastrus_orbiculatus_label.txt', 'Centaurea_stoebe_label.txt',
               'Cirsium_arvense_label.txt', 'Cirsium_discolor_label.txt', 'Geranium_maculatum_label.txt',
               'Geranium_robertianum_label.txt', 'Hemerocallis_fulva_label.txt', 'Hibiscus_moscheutos_label.txt',
               'Impatiens_capensis_label.txt', 'Iris_pseudacorus_label.txt']
label_names = ['Anemone_canadensis_label.txt']
val_file = 'patch_labels/Anemone_canadensis_label.txt'
IMAGENET_MEAN = [123.68, 116.779, 103.939]
# Learning params
learning_rate = 0.001
num_epochs = 500
batch_size = 25
# Network params
dropout_rate = 0.5
num_classes = 4
train_layers = ['fc8', 'fc7', 'fc6']
patch_size = 227
def main(argv):
    # read patches
    for name in label_names:
        tf.reset_default_graph()
        val_file = "patch_labels/" + name
        img_paths = []
        labels = []
        with open(val_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                specie = items[0].split(".")[0]
                filename = "patches/" + specie + "/" + items[0]
                # check for none type because of some broken data
                img = cv.imread(filename)
                if img is not None:
                    img_paths.append(filename)
                    labels.append(int(items[1]))
        length = 0
        batch_size = len(img_paths)
        print("Number of testing for " + name + ": " + str(batch_size))
        batch = []
        # load patches
        while length < len(img_paths):
            tiled_patches = np.ones((batch_size, patch_size, patch_size, 3), dtype=np.float32)
            remaining = (len(img_paths) - length) if (len(img_paths) - length) <= batch_size else batch_size
            for i in range(0, remaining):
                img = cv.imread(img_paths[length + i])
                img = cv.resize(img, (patch_size, patch_size))
                img = np.subtract(img, IMAGENET_MEAN)
                tiled_patches[i] = img
            length += batch_size
            batch.append(tiled_patches)

        # TF placeholder for graph input and output
        x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        keep_prob = tf.placeholder(tf.float32)
        # Initialize model
        model = AlexNet(x, keep_prob, num_classes, train_layers)

        # Link variable to model output
        score = model.fc8

        with tf.name_scope("predict"):
            predict = score
        predictions = []
        # Start Tensorflow session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print('Restoring weights from: ' + weights_file)
            saver = tf.train.Saver()
            saver.restore(sess, weights_file)
            for item in batch:
                prediction = sess.run(predict, feed_dict={x: item, keep_prob: 1.})
                pred_label = np.argmax(prediction, axis=1)
                predictions.append(pred_label)
        array = predictions[0]
        for i in range(1, len(predictions)):
            array = np.concatenate((array, predictions[i]), axis=0)
        array = array[0:len(labels)]
        # evaluate overall accuracy
        correct = sum((array == labels).astype(np.int))
        accuracy = correct / len(labels)
        print("Overall accuracy: " + str(accuracy))
        # evaluate non-background accuracy
        count = 0      # number of non background batch
        correct = 0     # correct prediction
        # number of feature predicted
        bud = 0
        flower = 0
        fruit = 0
        background = 0
        for i in range(0, len(array)):
            # not a background
            if(labels[i] != 3):
                count += 1
                if (array[i] == 0):
                    bud += 1
                elif (array[i] == 1):
                    flower += 1
                else:
                    fruit += 1
                if(array[i] == labels[i]):
                    correct += 1
            else:
                background += 1
        non_back_accuracy = correct / count
        print("Non-background accuracy: " + str(non_back_accuracy))
        print("Number of bud found: " + str(bud))
        print("Number of flower found: " + str(flower))
        print("Number of fruit found: " + str(fruit))
        print("Number of background found: " + str(background))
    return 0

if __name__ == "__main__":
    main(sys.argv)
