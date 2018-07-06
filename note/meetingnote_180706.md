# Meeting note 7/6
- Mentee: Jaewook Kang (@jwkanggist)
- Mento: Jihoon Jung


## 지난 Issues
- 크라우드웍스에 이어폰/헤드폰/모자 착용 사진 요청하기 (done)
- 크라우드웍스에 중간 수집 결과 확인 요청하기 (done)
- preprocessor 구현 (done)
- ~SSD로 bounding box detection 하는 모듈 붙이기 (연산량 문제로 일단 페)~
- gcp 사용법 익히기 (done)
- 모델 두번째 버전 완성하기 (doing)
- Data input pipeline 순서 변경 (doing)
- 크라우드웍스 Dataset 구매 (1008장) + 검수 완료 (done)
- 대도서관 + 봇노잼 유투부 영상 데이터 셋으로 만드는 모듈 구현 
- Pose classificaiton (거북목 판단)을 Deeplearning으로 해야하는지 Rule-based로 해야하는지 정해야함 --> (미해결)


## 진행 사항 요약
#### 1) 데이터 수집 관련
- 테스트 셋은 계속 짬짬히 모으는 중 [(구글 드라이브 링크)](https://drive.google.com/drive/folders/1_nvLmyYTc59l_1lYNUkvYVW3FyL0LkEy)
- 크라우드 데이터 셋 수집 완료 [(구글 드라이브 링크)](https://drive.google.com/open?id=1zq-j6DYQoPP4qleFlTohvty1jkMeenQY) 
    - 모자 + 헤드폰 데이터가 별로 없음 자체적으로 찍어서 보충예정

#### 2) 데이터 포맷 관련
- height x width = 1 x 1을 사용하고 모바일에서 resize하는 것으로 결정


#### 3) 첫번째 프로토 타입 TF model tflite 변환에 실패 
- 현재 r1.8의 `toco_converter`에 slim.batch_norm 변환에 문제가 있는걸로 판명되어 r1.9를 기다르는중 

#### 4) Google Cloud Platform을 처음 사용해 보는 데 진입 장벽이 좀 있음
- resnet example로 수행 
- [repo로 정리](https://github.com/jwkanggist/tpu-resnet-tutorial)

#### 5) Tensorflow conv + deconv 모듈 단위 테스트 완료 
- shape / name / 구조 정상확인 완료
- tflite 호환성 확인은 아직 


#### 6) 모바일앱 프로토 타입 구현 완료
- 프로토 타입 데모앱 구현 완료 (오픈소스로 구현된 모델 사용)
    - iOS : [(링크)](https://github.com/tucan9389/PoseEstimation-CoreML/tree/373ff20c77c4facd50a8b911a7a681a632d7bfe0)
    - Android: [(링크)](https://github.com/motlabs/mot-android-tensorflow/tree/develop/demo_pose-est)


## Issues
- 데이터 input pipeline 은 tfrecord --> augmentation --> data_loader의 순서임
- 모바일 동영상의 smooth한 동작을 위해서는 25fps정도 필요함 확인 (40ms inference time)
    - 현재 구글 픽셀폰에서 2.5fps, 400ms (우리 모델 아님)
- 연구 이슈

```bash
1) "거북목 판단"에서는 single hourglass안의 layer의 개수를 줄여야 하는가 아니면, stacked hourglass의 개수를 줄여야 하는가?

2) 하나의 single hourglass안에서의 레이어의 개수와
전체 hg layer stacking의 개수는 다르게 영향 할 것이고 각각 어떻케 다른 역할을 하는지 알아야 한다. 

—> stacked hourglass 의 개수는 중간 heatmap의 정도를 보고 판단할 수 있다.
—> hourglass모델안에서의 레이어의 개수는 추출하고자하는 part의  scale에 연관되어 있다. 작은 part를 추출하려고 할수록 레이어의 개수를 늘려야 한다. 

3) 계산량을 줄여야 한다
--> quantization (1/4)
--> channel 수 줄이기
--> hourglass layer수 줄이기
--> ~sparse 1x1 conv?~
```

## TODO (이번주)
- end to end training 코드 완성
- 훈련해보기 
- 유투브 영상으로 데이터 셋 만들기
- pose estimation benchmark 지표 조사

## 다음 미팅
- 
