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

insert demo gif 
![alt text]()

### Release Benchmarks
- Pose Estimation Accuracy (PCKh):  %

| Version | Framework            |  Device           | size (MB) | avg FPS |
|---------|----------------------|-------------------|-----------|---------|
| 1.0.0   | Android Pie + Tflite | Google Pixel2     |  1.4MB    |   TBU   |
| 1.0.0   | iOS 11.4.1  + CoreML | iPhoneX           |  1.5MB    |   TBU   |

### Repository Components
```bash
.
├── images              # some images for documentation
├── dataset/coco_form   # Unzip the dontbeturtle dataset at ./dataset/coco_form
├── note                # Some notes under Google Camps
├── sh_scripts          # A collection of shell scripts for easy operations
├── release             # dontbe turtle tflite and mlmodel here
└── tfmodules           # A collection of TF python files
```


## Mobile Frameworks


### How to Get .tflite and .mlmodel
```bash
$ python gen_tflite_coreml.py  --is-summary=False --import-ckpt-dir=<ckpt path directory>
# Args:
#  1) is-summary==True : collect tf summary for model graph
#     is-summary==False: None
#  2) --import-ckpt-dir: global path directory .ckpt stored
#
# An example:
# python gen_tflite_coreml.py  --is-summary=False --import-ckpt-dir=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules/export/model/run-20180815075050/
#
```

### Mobile Apps 
- Android repo : TBU 
- [iOS repo](https://github.com/motlabs/dont-be-turtle-ios) 


## Tensorflow Training Frameworks

### Technical Stacks
- Tensorflow (+ Tf Slim) >= 1.9
- Tf plot       == 0.2.0.dev0 
- opencv-python >= 3.4.2
- pycocotools   == 2.0.0
- Cython        == 0.28.4
- tensorpack    == 0.8.0
- tfcoreml      == 0.2.0

### Installation

- 1) Repository setup
```bash
$ git clone https://github.com/motlabs/dont-be-turtle
# cd dont-be-turtle/
$ git init
$ git submodule init
$ git submodule update

$ pip install -r requirement.txt
$ ./sh_scripts/install_tensorflow_gpu.sh
```



### How to Run Training
```bash
export MODEL_BUCKET=./tfmodules/export/model/       # set path for exporting ckpt and tfsummary
export DATA_BUCKET=./dataset/coco_form/dontbeturtle # set path for placing dataset
export SOURCE=./tfmodules/trainer_gpu.py            # set path for tensorflow trainer

python ${SOURCE}\
  --data_dir=${DATA_BUCKET}\
  --model_dir=${MODEL_BUCKET}\
  --is_ckpt_init=False\
  --ckptinit_dir=None
```
- You have an option to use `./sh_scripts/run_train_gpu.sh` with some custormization

> You need to create `dataset/coco_form/dontbeturtle/` and download our dataset from [here]().


## Dataset
- Training set (>15000)
    - [Youtubepose](https://www.robots.ox.ac.uk/~vgg/data/pose/) (4000)
    - [Sampled Shortbbcpose](https://www.robots.ox.ac.uk/~vgg/data/pose/) (7000)
    - [FLIC_train](https://bensapp.github.io/flic-dataset.html) (4000)
    - Custom  (216+462)

- Evaluation set (2000)
    - [FLIC_eval](https://bensapp.github.io/flic-dataset.html) (1000)
    - [Youtubepose](https://www.robots.ox.ac.uk/~vgg/data/pose/) (1000)

- Keypoint annotator repos
    - [For iOS Mobile](https://github.com/motlabs/KeypointAnnotation)
    - [For OSX](https://github.com/motlabs/dont-be-turtle-pose-annotation-tool)


## Benchmarks upto Model Config
- Percentage of correct keypoint (PCKh)
- Tflite model size (MB)
- Frame per sec (FPS) on Google Pixel 2

#### Baselines Model
- [MobileNet v2](https://arxiv.org/abs/1801.04381)
- [Stacked Hourglass](https://arxiv.org/abs/1603.06937)

#### wrt Number of HG stacking
- TBU 

| # of HG stages  |  # of HG stacking |  PCKh (%)  | tflite size (MB) | avg FPS |
|-----------------|-------------------|-----------|-------------------|---------|
| 4               |  4                |           |                   |         |  
| 4               |  2                |           |                   |         |
| 4               |  1                |           |                   |         |


#### wrt Number of HG stages
| # of HG stages  |  # of HG stacking |  PCKh (%)  | tflite size (MB) | avg FPS |
|-----------------|-------------------|-----------|-------------------|---------|
| 4               |  2                |           |                   |         |          
| 3               |  2                |           |                   |         |              
| 2               |  2                |           |                   |         |             



## Related  Materials
- [Jaewook Kang, "_From NIN to Inception V3_," Modulabs Machine Learning of Things (MoT) Lab 2018 Mar](https://docs.google.com/presentation/d/1JfH6bHnx14zlclglhoGIymzp0HJDQgE7g4gFKbudmkc/edit#slide=id.p3)
- [Jaewwok Kang, "_Machine Learning on Your Hands: Introduction to Tensorflow Lite Preview_," Tensorflow dev Exteneded X Modulabs, 2018 Apr](https://www.slideshare.net/modulabs/machine-learning-on-your-hand-introduction-to-tensorflow-lite-preview)
- [Jaewook Kang, "_Mobile Vision Learning: Model Compression and Efficient Convolution perspective_," ETRI, 2018 June 12th](https://docs.google.com/presentation/d/1_spnxEttqiTTh31c8S7xvHoSdZ3k4Rhm1f7GM7wNMdw/edit#slide=id.p1)
- [Jaewook Kang, "_Let's use Cloud TPU_", Aug 2018](https://docs.google.com/presentation/d/1LqlZc8IjXzp255UIXWQRBRGvvqwnLzkz1qAoq5YD1hs/edit?usp=drive_web&ouid=105579430994700782636)


## Project Contributors
- Dontbeturtle v1.0
    - [Jaewook Kang](https://github.com/jwkanggist/) (PI)
    - [Doyoung Gwak](https://github.com/tucan9389/)
    - [Jeongah Shin](https://github.com/Jeongah-Shin)
    - [YongGeunLee](https://github.com/YongGeunLee)
    - [Joonho Lee](https://github.com/junhoning)


## Acknowledgement
- This project was supported by Google Deep Learning Camp Jeju 2018.