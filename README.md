# Predicting bounding box with MNIST images

## Overview

This project is designed to predict the bounding boxes of digits within MNIST images using a deep learning approach. We have implemented different MLP and Convolutional Neural Network (CNN) to predict the coordinates of bounding boxes surrounding the digits in the images. This project is a continuation of my assignment 1 of the Deep Learning 2023 course at Bern University.

## Project Structure

The project is structured as follows:

-   mnist/ \# Contains the MNIST dataset
-   code/ \# Contains the main Jupyter notebook, model configurations, IOU function, and training functions
-   README.md \# Project overview and instructions

## Dataset

The dataset used for this project is the well-known MNIST dataset, which consists of grayscale images of handwritten digits. Each image is of size 28x28 pixels.

## Model Architecture

You can find the implementations of different model architectures in the model configurations Python files located in the code/ directory. The best-performing model is a CNN with the following specifications:

``` python
Conv_Model(
  (features): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (1): ReLU(inplace=True),
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (3): ReLU(inplace=True),
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (6): ReLU(inplace=True),
    (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (8): ReLU(inplace=True),
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    (10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (11): ReLU(inplace=True),
    (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    (13): ReLU(inplace=True),
    (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
  ),
  (classifier): Sequential(
    (0): Linear(in_features=1152, out_features=1024, bias=True),
    (1): ReLU(inplace=True),
    (2): Linear(in_features=1024, out_features=512, bias=True),
    (3): ReLU(inplace=True),
    (4): Linear(in_features=512, out_features=4, bias=True),
  )
)
```

## Evaluation

The metric used for evaluating the models is the Intersection Over Union (IoU). For two rectangles R1 and R2, it is defined as:

![IoU Equation](https://latex.codecogs.com/svg.image?\large&space;\bg%7Bwhite%7D&space;IoU(R_1,R_2)=\frac%7BA(R_1\cap&space;R_2)%7D%7BA(R_1\cup&space;R_2)%7D%7B\color%7BRed%7D%7D)

where A represents the area of the argument.
