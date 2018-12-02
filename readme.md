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

![alt text](https://github.com/MachineLearningOfThings/dont-be-turtle/blob/develop/images/turtle_180829_edit.gif)
> Note that the above model in .gif was trained only by 865 custom dataset.

### Release Benchmarks
- Pose Estimation Accuracy (PCKh): TBU

| Version | Framework            |  Device           | size (KB) | 
|---------|----------------------|-------------------|-----------|
| 0.5.0   | Android Pie + Tflite | Google Pixel2     |  749 KB    | 
| 0.5.0   | iOS 11.4.1  + CoreML | iPhoneX           |  811 KB    | 

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



### Mobile Apps 
- [Android repo](https://github.com/motlabs/dont-be-turtle-android)
- [iOS repo](https://github.com/motlabs/dont-be-turtle-ios) 


##   Frameworks

### Technical Stacks
- Tensorflow (+ Tf Slim) >= 1.9
- Tf plot       == 0.2.0.dev0 
- opencv-python >= 3.4.2
- pycocotools   == 2.0.0
- Cython        == 0.28.4
- tensorpack    == 0.8.0
- tfcoreml      == 0.2.0

### Repository Installation 

```bash
git clone https://github.com/motlabs/dont-be-turtle
# cd dont-be-turtle/
git init
git submodule init
git submodule update

pip install -r requirement.txt
./sh_scripts/install_tensorflow_gpu.sh
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
- You have an option to use `./sh_scripts/run_train_gpu.sh` with some customization

### How to Get .tflite and .mlmodel
> Note that you need to configure ./tfmodule/model/model_config_released.py before executing the below command. 
```bash
python gen_tflite_coreml.py  --is-summary=False --import-ckpt-dir=<ckpt path directory>
# Args:
#  1) is-summary==True : collect tf summary for model graph
#     is-summary==False: None
#  2) --import-ckpt-dir: global path directory .ckpt stored
#
# An example:
# python gen_tflite_coreml.py  --is-summary=False --import-ckpt-dir=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules/export/model/run-20180815075050/
#
```


## Donbeturtle Dataset v1.0
> You need to create `./dataset/coco_form/` and place the data set 
- [Donbeturtle dataset v1.0 (trainset only, 865 images) download](https://drive.google.com/open?id=122v9ZyRn-MGhrv9pXiplsO0AVZQitiNY)


- Keypoint annotator repos
    - [For iOS Mobile](https://github.com/motlabs/KeypointAnnotation)
    - [For OSX](https://github.com/motlabs/dont-be-turtle-pose-annotation-tool)



#### Baseline Papers
- [MobileNet v2](https://arxiv.org/abs/1801.04381)
- [Stacked Hourglass](https://arxiv.org/abs/1603.06937)



## Related  Materials
- [Jaewook Kang, " Don't be turtle project beyond Google Camp," GDGDevFest 2018 Pangyo, 2018 Nov](https://docs.google.com/presentation/d/1fxgYB1DbFVbRz0d_hIuG9DxFtLrhorWEDzDTJlf6f6U/edit#slide=id.g473a2a4e39_1_46)
- [Jaewook Kang, "_From NIN to Inception V3_," Modulabs Machine Learning of Things (MoT) Lab 2018 Mar](https://docs.google.com/presentation/d/1JfH6bHnx14zlclglhoGIymzp0HJDQgE7g4gFKbudmkc/edit#slide=id.p3)
- [Jaewwok Kang, "_Machine Learning on Your Hands: Introduction to Tensorflow Lite Preview_," Tensorflow dev Exteneded X Modulabs, 2018 Apr](https://www.slideshare.net/modulabs/machine-learning-on-your-hand-introduction-to-tensorflow-lite-preview)
- [Jaewook Kang, "_Mobile Vision Learning: Model Compression and Efficient Convolution perspective_," ETRI, 2018 June 12th](https://docs.google.com/presentation/d/1_spnxEttqiTTh31c8S7xvHoSdZ3k4Rhm1f7GM7wNMdw/edit#slide=id.p1)
- [Jaewook Kang, "_Let's use Cloud TPU_", Aug 2018](https://docs.google.com/presentation/d/1LqlZc8IjXzp255UIXWQRBRGvvqwnLzkz1qAoq5YD1hs/edit?usp=drive_web&ouid=105579430994700782636)


## Project Contributors
- Dontbeturtle v0.5
    - [Jaewook Kang](https://github.com/jwkanggist/) (PI)
    - [Doyoung Gwak](https://github.com/tucan9389/)
    - [Jeongah Shin](https://github.com/Jeongah-Shin)
    - [YongGeunLee](https://github.com/YongGeunLee)
    - [Joonho Lee](https://github.com/junhoning)
    - DongSeok Yang


## Acknowledgement
- This project was supported by Google Deep Learning Camp Jeju 2018.