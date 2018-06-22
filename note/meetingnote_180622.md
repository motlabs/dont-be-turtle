# Meeting note 6/22
- Mentee: Jaewook Kang (@jwkanggist)
- Mento: Jihoon Jung


## 지난 Issues
- 앱을 구동시키는 환경에 대한 명확한 정의가 필요함 --> (해결)
- 데이터 셋을 모으는 것이 우선순위 높여서 진행되어야 함 --> (해결)
- Google pixel phone 2의 준비가 필요함 --> (해결)
- Pose classificaiton (거북목 판단)을 Deeplearning으로 해야하는지 Rule-based로 해야하는지 정해야함 --> (미해결)


## 진행 사항 요약
#### 1) 데이터 수집 관련
- 훈련 셋 (1000장)은 크라우드 워스에서 46만원에 구매 
- 테스트 셋은 짬짬히 모으는 중 [(구글 드라이브 링크)](https://drive.google.com/drive/folders/1_nvLmyYTc59l_1lYNUkvYVW3FyL0LkEy)
- 일단 한번 구현해보고 성능 안나오면 제주도 가서 참가자 설정 조금 바꿔서 모을 예정
- Augmentation 적용하면 데이터 수가 모자르지는 않을 것 같음 (추측)
- [관련 문서 링크](https://docs.google.com/document/d/1C1kp-qXud6xqhB2-cuPmA1_YvcLfVMbs7udzqNoq3Zk/edit)
 
#### 2) 데이터 포맷 관련
- 논문에서는 일반적으로 height x width = 1 x 1을 사용
- 모바일 카메라 비율은  height x width = 4 x 3임
- 훈련시킬 때 부터 데이터의 가로세로 비율을 맞춰야 하는가 고민중

#### 3) 첫번째 프로토 타입 TF model tflite 변환에 실패 
- opset의 호환성을 고려하지 못했음 [(관련문서)](https://docs.google.com/document/d/19ZiExIc-vdbGrauuRUPKDI_o3sbVTeYMifc_hzvULGA/edit#heading=h.av0bqkw4ssvd)
- 비전 api는 tf.slim의 잘 지원하는 것 같아서 tf.slim.api을 이용하여 주로 다시 구현 예정 [(관련문서)](https://docs.google.com/document/d/19ZiExIc-vdbGrauuRUPKDI_o3sbVTeYMifc_hzvULGA/edit#heading=h.av0bqkw4ssvd)

#### 4) Google Cloud Platform을 처음 사용해 보는 데 진입 장벽이 좀 있음
- 삽질 중인데 구글에 요청에서 가이드 문서나 reference가 될 수 있는 example을 받았으면 좋겠음

#### 5) Tensorflow 단위 테스트 + 테스트 코드를 짜는 좋은 방법이 있으면 가이드를 받고 싶음

#### 6) Bounding box 필요해 보임 


## Issues
- 데이터 셋이 모자른 것은 유투브 영상을 이용하자 ( ex) 대도서관 + 봇노잼)
- opencv로 video를 frame으로 만들어서 저장할 수 있음

```python
# input frame
   if args.video == "":
       cam_src = -1
   else:
       cam_src = args.video
   cap = cv2.VideoCapture(cam_src)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
   
   while True:
       flag, frame = cap.read()
```

- 여기서 opencv는 BGR로 extract하기 때문에 RGB로 다시 채널 순서를 변환해야한다. 이 과정을 수행하지 않으면 훈련할 때는 상관이 없으나 inference에서 문제가 된다. 

```python
if flag:
           in_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
-  opencv이외에서 misc / pil등의 이미지 라이브러리를 사용할 수 있다. 
    - misc brg2rgb변환이 필요 없음
    - pil 출력이 numpy array가 아닌 이슈있음
- 모바일에서 전처리 / 후처리가 많이 없게 해야한다. 
- single pose estimation 인데 프레임안에 두 사람(a,b)이 들어오는 경우 
사람a의 목과 사람b의 머리가 연결 될 수 있다.
- bounding box + classification도 방법
- aspect ratio를 가로세로 비대칭으로 하는 것은 conv 연산 결과에 영향을 줄 수 있음. 
- resize는 어짭히 한번 하기 때문에 모바일에서 추가되는 연산이 큰 부담이 아닐 것임 (더 고민 필요)
- data augmentation은 random으로 해야함 
- 데이터 input pipeline 은 tfrecord --> augmentation --> data_loader의 순서임
    - augmentation은 선형 변환이끼 때문에 연산비용이 크지 않음
    - 그리고 inference time에 영향을 주지 않음
    - augmentation / shuffling 결과는 반드시 확인하해야
- pose estimation에서 좌표가 나오면 한 좌표를 기준으로 다른 좌표를 normalization해서 넣기 (상대좌)
- 모듈 테스트는 출력 shape + over/underflow 확인하는 정도로 (조금더 고민 필요)

## TODO
- 크라우드웍스에 이어폰/헤드폰/모자 착용 사진 요청하기 (done)
- 크라우드웍스에 중간 수집 결과 확인 요청하기 (done)
- Data input pipeline 순서 변경
- 대도서관 + 봇노잼 유투부 영상 데이터 셋으로 만드는 모듈 구현 
- preprocessor 구현
- SSD로 bounding box detection 하는 모듈 붙이기
- gcp 사용법 익히기
- 모델 두번째 버전 완성하기
- 

## 다음 미팅
- 6/29 금요일 오후 3시 30분 
