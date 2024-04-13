# DenseNet
## DenseNet Implementation with Tensorflow-Keras

![alt text](https://th.bing.com/th/id/R.9bc2e97eb9cd4be6d5502f208a5f5e57?rik=YUDpAYqxzlKRog&pid=ImgRaw&r=0)



## Overview
This repository contains the implementation of DenseNet, a deep learning model described in the paper "Densely Connected Convolutional Networks". The project includes a Python script and a Jupyter notebook that was developed in Google Colab.

## Files
1. `DenseNet.py`: This file contains the Python script for the project.
2. `DenseNet.ipynb`: This Jupyter notebook contains the process of creating and training the DenseNet model from scratch using Keras and TensorFlow.

## Dataset
The model is trained on 10,000 training images and tested on 1,000 test images from the CIFAR10 dataset.

## Model
The DenseNet model was trained for 5 epochs with a batch size of 32 using the Adam optimizer. The training accuracy achieved was 62%, and the validation accuracy was 58%.
The architecture of it includes 3 dense-blocks with k=4 batchnorm-relu-conv layers , each one following transition layer with batchnorm-relu-conv and a avgpool layer

## Future Work
The model's performance can be improved by training for more epochs, using the whole CIFAR10 dataset with 50000 train images, or using data augmentation techniques.

## Requirements
- TensorFlow
- Keras

## References
This project is based on the paper ["Densely Connected Convolutional Networks"](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html). 
