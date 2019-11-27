# Gender_detection
 This Module is from scratch using deep learning with keras, cvlib and opencv.

The keras model is created by using Keras Sequential model from scratch and training it on around 2200 images(1100 on each class). Face region is cropped by applying face-detection using cvlib on the images of the dataset. It achieved around 99.8% train accuracy and 94% test accuracy(20% of the dataset is used in testing over the model

# Python Packages
1. Tensorflow==1.14.1
2. numpy
3. opencv-contrib-python
4. TFLearn
5. Keras
6. progressbar
7. cvlib
8. Pillow
9. tf-nightly
10. requests
11. matplotlb
12. scikit-learn
install the following packages on your machine and be ready to tackle some errors.

### Note: Python 2.x is not supported

## Usage
### For Training the Model
python gendermodel.py --mode train 

### for WebCam feed
python Result_ModelWebcam.py

### from IP Webcam which is the android application to take the video feed remotely

python Result_Model.py

## Sample Output:


## Datset Used
#### Dataset is collected from online google images and is being shared in this repository. Dataset consist of 2200 images (each class has 1100 images). You can download the dataset and the pre-trained models from the given link:
https://drive.google.com/open?id=1C8mgXVRtx8DisoK_FV39HTxJMM5ZsLNV

## Pre Trained Model
#### You can download the pre trained model from this link 
https://drive.google.com/open?id=1GlTEWYjKT3dsRkELiGMn0f89vU2pNGrh


