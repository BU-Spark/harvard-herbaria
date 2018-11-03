# YoloNet for Plant Specimen Feature Detection
Plant speciemen feature detection using YOLOnet. We provide pre-trained weights from training on images from 14 categories:
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

Code includes training, predicting for one image and evaluate the accuracy for all 14 models on training sets.
Original work from https://github.com/hizhangp/yolo_tensorflow
## Installation Prerequisite
1. Linux system (I used Ubuntu 16.04, have not tested the code on any other Linux distributions)
2. Python 3.6, Tensorflow r1.10, OpenCV 3.4.2, CUDA(optional, you don't really need this for prediction and model evaluation) 
## Installation
1. Clone the repository to local. (the source code)
2. Download the image set from https://drive.google.com/open?id=1qeQfS7czqIF7jhl67X1MrKpIU6f4TgwE and put the unzipped folder to the same
   folder as the source code.(.zip file is 2GB)
3. Donwload the pre-trained weights from https://drive.google.com/open?id=1OTo3T4aP3nobsbpWF6hLmM2yTflDFeQH and put the unzipped folder to
   the same folder as the source code.(.zip file is 4.5GB)
## Run the model
We provide two way of using YoloNet:
### Predict the features in any given image.
If you have a filename to an image and its category name, use predict.py. Note that the category name has to be exactly the same as the
ones listed above.
For example if you have a image named Anemone_canadensis_1111.jpg and the category is Anemone Canadensis, enter the following line:
```
python predict.py Anemone_canadensis_1111.jpg Anemone_canadensis
```
![alt text](https://github.com/BU-Spark/harvard-herbaria/blob/siqi/example/Anemone_canadensis.1040272.17269.jpg)
The output file will be an image name as "output.jpg" with the feature bounded by colored square:
 - red: bud
 - green: flower
 - blue: fruit
 ### Evaluate the model
 You can also evaluate all the accuracies for all the models on the training sets. Run the following line:
```
python eval_model.py
```
which will compute the category-wise average accuracies, visualize the detection result saving them to local folder named "predictions" , model by model.
