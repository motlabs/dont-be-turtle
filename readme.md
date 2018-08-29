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


![alt text]()


### Baselines Model
- [MobileNet v2](https://arxiv.org/abs/1801.04381)
- [StackedHourglass](https://arxiv.org/abs/1603.06937)


### Mobile Apps 
- Android repo
- iOS repo


## Benchmarks
- Percentage of correct keypoint (pck)
- Tflite model size (MB)
- Frame per sec (FPS) over Google Pixel 2


### wrt Number of HG stacking
| # of HG stages  |  # of HG stacking |  pck (%)  | tflite size (MB) | avg FPS |
|-----------------|-------------------|-----------|------------------|---------|
| 4                 4           
| 4                 2           
| 4                 1           


### wrt Number of HG stages
| # of HG stages  |  # of HG stacking |  pck (%)  | tflite size (MB) | avg FPS |
|-----------------|-------------------|-----------|------------------|---------|
| 4                 2           
| 3                 2           
| 2                 2           



## Tensorflow Training Frameworks

### Technical Stacks
- Tensorflow >= 1.9
- Tf slim 
- Android + Tflite 
- iOS + CoreML 



```bash
.
├── dataset/coco_form   # Unzip the dontbeturtle dataset at ./dataset/coco_form
├── note                # Some meeting notes under Google Camps
├── sh_scripts          # A collection of shell scripts for easy training 
└── tfmodules           # A collection of TF python files
```

### Run training
```bash
export MODEL_BUCKET=./tfmodules/export/model/       # set path for export ckpt and tfsummary
export DATA_BUCKET=./dataset/coco_form/dontbeturtle # set path for dataset
export SOURCE=./tfmodules/trainer_gpu.py            # set path for tensorflow trainer

python ${SOURCE}\
  --data_dir=${DATA_BUCKET}\
  --model_dir=${MODEL_BUCKET}\
  --is_ckpt_init=False\
  --ckptinit_dir=None
```


### Dataset
- Training set (>15000)
    - Youtubepose (4000)
    - Sampled Shortbbcpose (7000)
    - FLIC_train (4000)
    - Custom  (216+462)

- Evaluation set (2000)
    - FLIC_eval (1000)
    - Youtubepose (1000)


## Related  Materials
- [Jaewook Kang, "_From NIN to Inception V3_," Modulabs Machine Learning of Things (MoT) Lab 2018 Mar](https://docs.google.com/presentation/d/1JfH6bHnx14zlclglhoGIymzp0HJDQgE7g4gFKbudmkc/edit#slide=id.p3)
- [Jaewwok Kang, "_Machine Learning on Your Hands: Introduction to Tensorflow Lite Preview_," Tensorflow dev Exteneded X Modulabs, 2018 Apr](https://www.slideshare.net/modulabs/machine-learning-on-your-hand-introduction-to-tensorflow-lite-preview)
- [Jaewook Kang, "_Mobile Vision Learning_," Hanlim Univ, 2018 May](https://www.slideshare.net/JaewookKang1/180525-mobile-visionnethanlimextended)
- [Jaewook Kang, "_Mobile Vision Learning: Model Compression and Efficient Convolution perspective_," ETRI, 2018 June 12th](https://docs.google.com/presentation/d/1_spnxEttqiTTh31c8S7xvHoSdZ3k4Rhm1f7GM7wNMdw/edit#slide=id.p1)



## Project Contributors
Modulabs, Machine Learning of Things (MoT) labs members:
- *[Jaewook Kang](https://github.com/jwkanggist/)
- [Doyoung Gwak](https://github.com/tucan9389/)
- [Jeongah Shin](https://github.com/Jeongah-Shin)
- [YongGeunLee](https://github.com/YongGeunLee)
- [Joonho Lee](https://github.com/junhoning)


## Acknowledgement
- This project was supported by Google Deep Learning Camp Jeju 2018.