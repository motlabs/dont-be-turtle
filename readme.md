# Don't be a Turtle Project

Author: Jaewook Kang



## About

### Project objectives

This Don’t be a Turtle Project makes all of IT people have right posture and feel good while they are working! 
We investigate a mobile machine learning based methodology providing 
feedbacks with respect to your neck posture. 
For this purpose, we monitor neck, detecting 
whether you are maintaining good working posture. 
If you are working in an overhanging posture, you will be alerted to maintain a good posture.


![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/develop/images/about.jpg)


### Our Solution Approach
- A Classification + estimation approach
    - Pose Estimation Task: Neck pose estimation from CNN features using four joint positions of human body: head, neck, right shoulder, left shoulder
    - Pose Classification task: Classification whether neck posture is neck tech from CNN features.


![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/develop/images/approach.jpg)


## Keywords
- Tech neck classification
- Human pose estimation
- Transfer learning
- Mobile convolutional neural networks
- Tensorflow Lite

## Technical Stacks
- Tensorflow
- Tf slim library (python model module building, pb/ckpt file export)
- Tensorflow lite
- Android + nnapi / iOS + coreML (Mobile running optimization and Hardware delegation)


## Expected Results
![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/develop/images/product.jpg)

### Product outputs
- Tensorflow model (pb/ckpt)
- Tensorflow lite model (tflite)
- An Android/iOS Mobile benchmark APP
- An arXiv Paper



### Benchmarks
- Million Mult-Add / Parameters
- tflite model size (MB)
- per-runtime accuracy (acc/ms) ([see LPIRC CVPR 2018 measure](https://docs.google.com/document/d/1_toBzIrfcrZwxF9B1jMIbMvqxrw9AS1rWy-fdSP_OvI/edit))
- App Battery consumption (mAh)



## What we do in Jeju Camp?
In the camp, we aim to mainly focus on the below items:
- Reducing model size
- Reducing inference time (App battery consumption)
- Improving classification accuracy.
- Checking feasibility of transfer learning, whether face detection data sets are effective to neck pose estimation.

Most of development works and background study will be done before starting the Jeju camp.
- Mobile CNN background study (On going)
- Pose estimation background study  (On going)
- App implementation  (On going)
- Data set labeling and managing python module
- Tensorflow model training / validation framework development
- ~Tflite conversion (Done)~


## Tentative Schedules
~~- Apr: Writing project proposal and submission~~

~~- May: Background study and establishing a baseline model using mobilenetv2 and DeepPose ideas~~

- June:
    - Tensorflow development to shorten training pipeline.
    - Tensorflow to Tensorflowlite conversion automation
    - Building a benchmark android or iOS Apps.

- July (In Jeju camp)
    - Week1: Investigation for improving accuracy of our proposed model without concerning model size and inference time.
    - Week2: Investigation for reducing model size while maintaining the accuracy
    - Week3: Investigation for reducing inference time given maintaining the accuracy and the model size.
    - Week4: Paper writing and final presentation preparation


## Baselines


### Dataset Baseline
- [FLIC dataset 4000+1000 (training + vtest)](https://bensapp.github.io/flic-dataset.html)
- [LSP  dataset 11000+1000 (training +test)](http://sam.johnson.io/research/lsp.html)
- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#)
    - Image dataset
    - 25K images
    - 410 types of activities
    - 2D annotation
- https://posetrack.net/
    - 500 video sequence → 20k frames
    - 2D annotation
- [VGG pose dataset](https://www.robots.ox.ac.uk/~vgg/data/pose/)
    - YouTobe pose
    - BBC pose
    - BBC extended pose
    - Short BBC pose
    - 2D annocation

- Also see [this](https://docs.google.com/document/d/1C1kp-qXud6xqhB2-cuPmA1_YvcLfVMbs7udzqNoq3Zk/edit#)


### Model Baselines
- Mobile CNN models
    - [Mobilenet v1](https://arxiv.org/abs/1704.04861)
    - [Mobilenet v2](https://arxiv.org/abs/1801.04381)
    - [SqueezeNet](https://arxiv.org/abs/1602.07360)
    - [Shufflenet](https://arxiv.org/abs/1707.01083)
    - [Unet](https://arxiv.org/abs/1505.04597)

- Pose estimation models
    - [DeepPose](https://arxiv.org/abs/1312.4659)

## Related  Activities
- [Jaewook Kang, "_From NIN to Inception V3_," Modulabs Machine Learning of Things (MoT) Lab 2018 Mar](https://docs.google.com/presentation/d/1JfH6bHnx14zlclglhoGIymzp0HJDQgE7g4gFKbudmkc/edit#slide=id.p3)
- [Jaewwok Kang, "_Machine Learning on Your Hands: Introduction to Tensorflow Lite Preview_," Tensorflow dev Exteneded X Modulabs, 2018 Apr](https://www.slideshare.net/modulabs/machine-learning-on-your-hand-introduction-to-tensorflow-lite-preview)
- [Jaewook Kang, "_Mobile Vision Learning_," Hanlim Univ, 2018 May](https://www.slideshare.net/JaewookKang1/180525-mobile-visionnethanlimextended)
- Jaewook Kang, "_Mobile Vision Learning: Model Compression perspective_," ETRI, 2018 June (TBU)


## Further Application Extension
- Distracted driver detection ([A Kaggle link](https://www.kaggle.com/c/state-farm-distracted-driver-detection#description))
    - Which is not in the scope of the Jeju camp

## Project Contributors
Modulabs, Machine Learning of Things (MoT) labs members:
- 2018 June: Jaewook Kang, Sungjin Lee, Seoyeon Yang, Joonho Lee, Yunbum Baek, Joongwon Hwang, Doyoung Gwak, Jeongah Shin, Taekmin Kim, YongGeunLee, Jihwan Lee, Jonguk Lee