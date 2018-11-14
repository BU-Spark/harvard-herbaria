# Alexnet with sliding window for Plant Specimen Feature Detection
Plant speciemen feature detection using alexnet. The network is trained based on mini-patches of specimen feature generated from
the original images. Here is the list of species:
 - Anemone_canadensis
 - Anemone_hepatica
 - Aquilegia_canadensis
 - Bidens_vulgata
 - Celastrus_orbiculatus
 - Centaurea_stoebe
 - Cirsium_arvense
 - Cirsium_discolor
 - Geranium_maculatum
 - Geranium_robertianum
 - Hemerocallis_fulva
 - Hibiscus_moscheutos
 - Impatiens_capensis
 - Iris_pseudacorus'

Code includes training, predicting for one image and evaluate the accuracy with testing sets.
Original work from https://github.com/kratzert/finetune_alexnet_with_tensorflow
## Installation Prerequisite
1. Linux system (I used Ubuntu 16.04, have not tested the code on any other Linux distributions)
2. Python 3.6, git Tensorflow r1.10, OpenCV 3.4.2, CUDA(optional, you don't really need this for prediction and model evaluation) 
## Installation
1. Clone the repository to local. (the source code)
2. Download the minipatch set from https://drive.google.com/open?id=1P6mAUsI8rawqI5I40qPG1hJtmGclReKJ and put the unzipped folder to the same
   folder as the source code.
3. Donwload the pre-trained weights from https://drive.google.com/open?id=1-ez84pxwSv--An1-GZEu3lH7xhmwqGWZ and put the unzipped folder to
   the same folder as the source code.
## Run the model
We provide three ways of using Alexnet
### Predict the features in any given image.
If you have a filename to an image, use predict.py. 
```
python predict.py (image_path)
```
For example if you have a image named Anemone_canadensis_1111.jpg and the category is Anemone Canadensis, enter the following line:
```
python predict.py Anemone_canadensis_1111.jpg
```
The output file will be an image name as "output.jpg" with the feature bounded by colored square:
 - red: bud
 - green: flower
 - blue: fruit
 ### Evaluate the model
 You can also evaluate minipatch prediction the accuracies a testing set:
```
python eval_model.py
```
which will compute the overall accuracies and non-background prediction accuracies
You can also use the model to predict a label for a minipatch, use test.py
```
python predict.py (minipatch_path)
```
