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

#### 4) Google Cloud Platform에 어느정도 익숙해
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
- 데이터 input pipeline 은 `tfrecord --> augmentation --> data_loader`의 순서임
    - data augmentation시 randomness를 포기 한다면 augmentation 이후에 tfrecord로 저장해 놓는것이 좋을 것이라는 구글 엔지니어(Sourabh)의 코멘트
    - step 마다 data augmentation의 randomness가 필요하고 그렇기 때문에 위의 구조 그대로 가는 것이 좋음
- 모바일 동영상의 smooth한 동작을 위해서는 25fps정도 필요함 확인 (40ms inference time)
    - 현재 구글 픽셀폰에서 2.5fps, 400ms (우리 모델 아님)
- 연구 이슈

```bash
1) "거북목 판단"에서는 single hourglass안의 layer의 개수를 줄여야 하는가 아니면, stacked hourglass의 개수를 줄여야 하는가?

2) 하나의 single hourglass안에서의 레이어의 개수와
전체 hg layer stacking의 개수는 다르게 영향 할 것이고 각각 어떻케 다른 역할을 하는지 알아야 한다. 

—> stacked hourglass는 전체 이미지의 맥락을 이해하는 것을 도와준다. 
—> stacked hourglass 의 개수는 중간 heatmap의 정도를 보고 판단할 수 있다.

—> hourglass모델 안에서의 레이어의 개수는 추출하고자하는 part의  scale에 연관되어 있다. 
다양한 스케일을 다룰 수 록 할수록 레이어의 개수를 늘려야 한다. 
즉 추상화 문제

3) output 레이어는 1x1 conv를 stacking해서 구성한다. 
-> 관련 논문을 좀 더 살펴본다. 
-> 일단 2 레이어를 stacking하는 것으로 시작한다. 

4) 거북목 classification 케이스
거북목 classifier는 pose estimation에서 아래를 넘겨 받는다. 
- 각 part의 est 좌표: head, neck, Rsholder, Lsholder
- 각 part 의 confidance level: 각 heatmap의 argmax값

케이스
- 1. 여러 사람일 때:  pose에서 주는 좌표 confidence 레벨로 거르기
- 2. 아무도 없을 때:  pose에서 주는 좌표 confidence 레벨로 거르기
- 3. 잘 못 찾았을 때(특정 좌표가 없을때):  pose에서 주는 좌표 confidence 레벨로 거르기 
- 4. 좋은 자세 일 때: 바른 자세
- 5. 나쁜 자세 일 때: 거북목 자세
- 6. 기타 자세 일 때(4개의 part가 다 주어지는 경우): pose에서 주는 좌표 confidence 레벨로 거르기 

classification group
- None: 1,2,3,6 
- GOOD: 4
- BAD: 5

5) 계산량을 줄여야 한다
-> quantization (1/4)
-> channel 수 줄이기
-> hourglass layer수 줄이기
-> ~sparse 1x1 conv?~

```
- 데이터 한번 훈련해보고 데이터 셋의 다양성이 부족하면 추가 수집한다. 

## TODO (이번주)
- end to end training 코드 완성
- 훈련해보기 
- 유투브 영상으로 데이터 셋 만들기
- pose estimation benchmark 지표 조사
- TF r1.9가 릴리즈 되는데로 tf.contrib.lite.TocoConvertor()사용해보기 
- 기타 TF 모델 meta parameter에 추가 할 것들
    - output layer stacking config

## 다음 미팅
- 지훈님께서 7/11 (수)에 제주도에 오심
