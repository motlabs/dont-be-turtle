# Don't be a Turtle Project

Author: Jaewook Kang


## Purpose

1. Establishing benchmarks for mobile CNN research
2. Providing new user experience
3. Simple Implementation

## About

### Description

This **Don't be a Turtle Project** makes all of IT people have right posture and feel good while they are working!
We investigate a mobile machine learning based methodology providing feedbacks with respect to your neck posture. 
For this purpose, we monitor neck, detecting  whether you are maintaining good working posture. 
If you are working in an overhanging posture, you will be alerted to maintain a good posture.

![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/develop/images/about.jpg)

### Our Solution Approach

- A Classification + estimation approach
  - **Pose Estimation task** : Neck pose estimation from CNN features containing four joint positions of human body: head, neck, right shoulder, left shoulder
  - **Pose Classification task** : Classification whether neck posture is neck tech from CNN features.
- Reference:
    - [Toshev and Szegedy, &quot;DeepPose: Human Pose Estimation via Deep Neural Networks&quot;, CVPR 2014](https://arxiv.org/abs/1312.4659)

![alt text](https://github.com/MachineLearningOfThings/smile-turtle-proj/blob/develop/images/approach.jpg)

### Keywords

- Tech neck classification
- Human pose Estimation
- Transfer learning
- Mobile Convolutional neural networks
- Tensorflow lite

### Technical Stacks

- [Tf slim library](https://github.com/tensorflow/models/tree/master/research/slim/nets) (python model module building, pb/ckpt file export)
- [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (pb/ckpt file export)
- Tensorflow lite conversion (tflite)
- Android  nnapi / iOS coreML

## Expected Results

### Products

- Tensorflow Tech model (pb/ckpt)
- Tensorflow lite model (tflite)
- Mobile benchmark APP
  - Android
  - iOS
- An arXiv Paper

### Benchmark

- tflite model size (MB)
- Millon mult-add
- Average Inference Time in App (ms)
- App Battery consumption (mAh)



## Research Baselines

### Tensorflow code 

- [SungKim smile-more repo](https://github.com/hunkim/smile_more)
- [Face-detection-with-mobilenet-ssd](https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd)
- [Tensorflow object detection api github](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Tf Slim.nets repo](https://github.com/tensorflow/models/tree/master/research/slim/nets)
- [Tensorflow lite support ops set](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md)
- [Pre-converted tflite model list](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)

### Mobile Apps

- Tensorflow lite demo:
    - Android: [Readme](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/demo_android.md), [codes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples/android)
    - iOS demo: [Readme](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/demo_ios.md), [codes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples/ios)
- [Tensorflow android camera demo (tensorflow lite를 사용하지 않고 tensorflow c++사용)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)

### Data set :

- Facial expression classification
    - [Affective-MIT facial expression data set (AM-FED),](https://www.affectiva.com/science-resource/affectiva-mit-facial-expression-dataset-am-fed/) [paper](https://www.affectiva.com/wp-content/uploads/2017/03/Crowdsourcing_Facial_Responses_to_Online_Videos._IEEE_Transactions_on_Affective_Comp.pdf)
    - [Kaggle facial expression dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Face detection
- [AFW dataset](https://www.ics.uci.edu/~xzhu/face/), [paper](https://www.ics.uci.edu/~xzhu/paper/face-cvpr12.pdf)
- [PASCAL dataset](http://host.robots.ox.ac.uk/pascal/VOC/databases.html)
- [FDDB dataset](http://vis-www.cs.umass.edu/fddb/), [paper](http://vis-www.cs.umass.edu/fddb/fddb.pdf)
- [Wider face dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), [paper](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/paper.pdf)

### CNN  Model 

- [Mobilenet v2](https://arxiv.org/pdf/1801.04381.pdf)
- [NasNet](https://arxiv.org/abs/1707.07012)
- [SqueezeNet](https://arxiv.org/abs/1602.07360)

### Human Pose Estimation  

- [DeepPose](https://arxiv.org/abs/1312.4659)
- [Tensorflow human pose estimation](https://github.com/ildoonet/tf-pose-estimation)

