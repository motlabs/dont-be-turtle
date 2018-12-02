# Meeting note 7/17
- Mentee: Jaewook Kang (@jwkanggist)
- Mento: Jihoon Jung


## 지난 Issues
- end to end training 코드 완성 (doing)
- TPU 훈련해보기   (doing)
- pose estimation benchmark 지표 조사  (done)
- TF r1.9가 릴리즈 되는데로 tf.contrib.lite.TocoConvertor()사용해보기   (done)
- 기타 TF 모델 meta parameter에 추가 할 것들 (done)
    - output layer stacking config (done)
    - hg layer stacking config (done)
    
    
## 진행 사항 요약
#### 1) 데이터 수집 관련
- LSP dataset annotation + label 포맷 변환 완료
- [BBC pose (40GB) / Youtube pose (85MB) 추가](https://www.robots.ox.ac.uk/~vgg/data/pose/) 
    - label 포맷 변환 아직

#### 2) trainer_tpu.py 구현
- 구글의 resnet_main.py을 baseline으로 구현
- 계속 다른 모듈을 붙이면서 확인중
    - data_loader_tpu.py 통합 확인
    - model_builder.py 통합확인
    - tpu을 돌리기 위해서는 모델의 모든 포인트에서의 shape가 구체화 되어 있어야 한다. (?은 안된다)


#### 3) tfrecord_convertor, data_loader_tpu 구현
- tfrecord 변환 확인
    - scalar는 _float_feature() / _int64_feature()로 저장
    - string and np.array는 _byte_feature()로 저장
    - tf.FixedLenFeature() 와 tf.VarLenFeature() 동작 차이 파악필요
- data_loader_tpu.py 
    - input_fn(): tf.data.Dataset.list_files() --> [tf.TFRecordDataset()](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) --> shuffle() --> tf.data.map(parser)-->tf.data.batch()
    - parser():  define keys_to_features --> [tf.parse_single_example](https://www.tensorflow.org/api_docs/python/tf/parse_single_example) (bytes)  
    --> tf.reshape (bytes) --> preprocessing (Tensor) --> return image and labels (Tensor)
    

#### 4) preprocessing 진행중
- preprocess_image()  
- preprocess_for_train() 
- _decode_and_random_crop() 
- distorted_bounding_box_crop()
- _heatmap_generator()


#### 5) Tflite 변환
- 전 모듈 동작확인
- neareset neighbor upsampler중 `tf.tile()`이 tflite과 호환되지 않음

#### 6) training pipeline 정리 
Raw_image (jpg/jpeg) → tfrecord_convertor → data_loader (tf.data) → preprocessing → return to trainer.py
Raw_label (json)     → tfrecord_converter → data_loader (tf.data) → heatmap_gen   → return to trainer.py

#### 7) TFCON oral 발표 
- 만족스럽지 않지만 일단 끝남
- 발표 연습 부족 + 영어 실력 부족 + 영어 발음 구림 ㅠ

## Issues

- [tf.image.sample_distorted_bounding_box](https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box)
- preprocessing의 random_crop을 위한 bounding box좌표를 리턴하는 함수
```
- 입력파라미터 
    - bounding_boxes:       기본 바운딩 박스. random cropping이 적용되는 범위. [x_min,y_min,x_max,y_max]로 표시. 
    - min_object_covered:   기본 바운딩 박스에서 cropped box가 겹쳐야 하는 범위 \in [0,1]
    - aspect_ratio_range:   random하게 생성되는 cropped box의 aspect ratio 범위 [0.85,1.15]             
    - area_range:           random하게 생성되는 cropped box가 입력 전체 이미지를 포함하는 비율의 범위                          
    - max_attempts:         random cropped box 생성 시도 횟수 10 사용                  
    - use_image_if_no_bounding_boxes = True: 이렇케 설정하면 bounding box가 전체 범위로 설정된다. 
- 리턴:
    begin:  A Tensor containing [offset_height, offset_width, 0]
    size:   A Tensor containing [target_height, target_width, -1]
    bboxes: A Tensor with shape [1, 1, 4] containing the distorted bounding box
```
    
- 문제 적용에서 고려해야할 사항
    - cropping 순서: image (orig) --> b) random_crop (?) --> d) resize(256)
    - keypoint가 잘리는 경우 어떻케 할 것인가?
    - label coordinate scaling
    
- 파라미터 설정    
    - 기본적으로 아래의 순서를 따름
    - 기본 bounding_boxes는 = [0,0,1,1] (전체 이미지)
    - min_object_covered  = 0.1  (디폴트)
    - aspect_ratio_range  = [0.85,1.15]
    - area_range          = [0.5 1] (전체 이미지에서 30%이상 포함하는 cropping)
    - max_attempts        = 10
    
질문1) heatmap 생성과정에서 point scaling작업
- 답변) 'crop_window = [offset_y,offset_x,target_height,target_width]'을 받아서 heatmap_generator에 넘겨줘서 scaling 가능
    - keypoint가 cropping안에 들어왔는지 확인
    - 만약 들어온경우 linear shift by [offset_y,offset_x]
    - aspect_ratio_height/width로 scaling
        - aspect_ratio_height = 256 / target_height 
        - aspect_ratio_width = 256 / target_width 

질문3) cropping할때 bounding box 크기를 어느정도로 잡는게 좋은지?
    - 전체 이미지에서 30%이상 포함하는 cropping

질문4) 논문에서는 rotation을 했다고 나오는데 어떻케 해야할까요 ?
- [tf.contrib.image.rotate 사용](https://www.tensorflow.org/api_docs/python/tf/contrib/image/rotate)
- +30 +15 0 -15 -30 (deg) 만 포함
- 우선순위 떨어짐
- 아래 함수 사용가능
```python
def random_rotation(rand_val, image):
   min_rad = -rand_val * np.pi / 180.0
   max_rad = rand_val * np.pi / 180.0
   # angle = tf.Variable(tf.random_uniform([1],min_rad, max_rad))
   rad = np.random.uniform(min_rad, max_rad)
   image = tf.contrib.image.rotate(image, rad, interpolation='BILINEAR')
   print('rr. angle: %.3f' % rad)

   return image
```

질문5) origianl image와 augment한 image가 concat해서 출력되어야 한느거 맞나요 ?
    - 답변) 아님. augumented image만 출력되면됨. epoch이 많아지면 자연스럽게 모든 정보가 포함됨


## TODO (이번주)
- preprocessor.py 완성
- GCP TPU로 훈련 돌리기
- 
    

## 다음 미팅
- 이슈 생기면 수시로
