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


![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/images/about.jpg)


### Our Solution Approach
- A Classification + estimation approach
    - Pose Estimation Task: Neck pose estimation from CNN features using four joint positions of human body: head, neck, right shoulder, left shoulder
    - Pose Classification task: Classification whether neck posture is neck tech from CNN features.


![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/images/approach.jpg)


### Keywords
- Tech neck classification
- Human pose estimation
- Transfer learning
- Mobile convolutional neural networks
- Tensorflow Lite

### Technical Stacks
- Tensorflow
- Tf slim library (python model module building, pb/ckpt file export)
- Tensorflow lite conversion (tflite)
- Android + nnapi / iOS + coreML (Mobile running optimization and Hardware delegation)


## Expected Results

###Product outputs
- Tensorflow model (pb/ckpt)
- Tensorflow lite model (tflite)
- An Android/iOS Mobile benchmark APP
- An arXiv Paper

![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/images/product.jpg)


### Benchmarks
- Million Mult-Add
- Million Parameters
- tflite model size (MB)
- per-runtime accuracy (acc/ms) ([LPIRC CVPR 2018 measure](https://docs.google.com/document/d/1_toBzIrfcrZwxF9B1jMIbMvqxrw9AS1rWy-fdSP_OvI/edit))
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
~~Apr: Writing project proposal and submission~~
~~May: Background study and establishing a baseline model using mobilenetv2 and DeepPose ideas~~
June:
Tensorflow development to shorten training pipeline.
Tensorflow to Tensorflowlite conversion automation
Building a benchmark android or iOS Apps.
July (In Jeju camp)
Week1: Investigation for improving accuracy of our proposed model without concerning model size and inference time.
Week2: Investigation for reducing model size while maintaining the accuracy
Week3: Investigation for reducing inference time given maintaining the accuracy and the model size.
Week4: Paper writing and final presentation preparation


Research Baselines
Tensorflow code baselines
SungKim smile-more repo
Face-detection-with-mobilenet-ssd
Tensorflow object detection api github
Tf Slim.nets repo
Tensorflow lite support ops set
Pre-converted tflite model list


Mobile Apps
Tensorflow lite demo:
Android: Readme, codes
iOS demo: Readme, codes
Tensorflow android camera demo (tensorflow lite를 사용하지 않고 tensorflow c++사용)

Data set Baseline:
Facial expression classification
Affective-MIT facial expression data set (AM-FED), paper
Kaggle facial expression dataset
Face detection
AFW dataset, paper
PASCAL dataset
FDDB dataset, paper
Wider face dataset, paper

CNN classification Model Baselines
Mobilenet v2
NasNet
SqueezeNet

Human pose estimation Model baseline
DeepPose
Tensorflow human pose estimation


Our Challenges
Collecting Dataset for tech neck pose estimation and classification
App battery consumption and TF model size
Classification accuracy when squeezing the TF model.
Tensorflow lite ops set may not fully support tensorflow ops used in the TF model implementation.
Further Application Extension
Distracted driver detection (A Kaggle link)
Which is not in the scope of the Jeju camp

Contributors
Modulabs machine learning of things (MoT) labs members:

