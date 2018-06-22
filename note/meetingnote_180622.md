# Meeting note 6/22
- Mentee: Jaewook Kang (@jwkanggist)
- Mento: Jihoon Jung


## 지난 Issues
- 앱을 구동시키는 환경에 대한 명확한 정의가 필요함 --> (해결)
- 데이터 셋을 모으는 것이 우선순위 높여서 진행되어야 함 --> (해결)
- Google pixel phone 2의 준비가 필요함 --> (해결)
- Pose classificaiton (거북목 판단)을 Deeplearning으로 해야하는지 Rule-based로 해야하는지 정해야함 --> (미해결)


## 진행 사항 요약
1) 데이터 수집 관련
- 훈련 셋 (1000장)은 크라우드 워스에서 46만원에 구매 
- 테스트 셋은 짬짬히 모으는 중 [(구글 드라이브 링크)](https://drive.google.com/drive/folders/1_nvLmyYTc59l_1lYNUkvYVW3FyL0LkEy)
- 일단 한번 구현해보고 성능 안나오면 제주도 가서 참가자 설정 조금 바꿔서 모을 예정
- Augmentation 적용하면 데이터 수가 모자르지는 않을 것 같음 (추측)
- [관련 문서 링크](https://docs.google.com/document/d/1C1kp-qXud6xqhB2-cuPmA1_YvcLfVMbs7udzqNoq3Zk/edit)
 
2) 데이터 포맷 관련
- 논문에서는 일반적으로 height x width = 1 x 1을 사용
- 모바일 카메라 비율은  height x width = 4 x 3임
- 훈련시킬 때 부터 데이터의 가로세로 비율을 맞춰야 하는가 고민중

3) 첫번째 프로토 타입 TF model tflite 변환에 실패 
- opset의 호환성을 고려하지 못했음 [(관련문서)](https://docs.google.com/document/d/19ZiExIc-vdbGrauuRUPKDI_o3sbVTeYMifc_hzvULGA/edit#heading=h.av0bqkw4ssvd)
- 비전 api는 tf.slim의 잘 지원하는 것 같아서 tf.slim.api을 이용하여 주로 다시 구현 예정 [(관련문서)](https://docs.google.com/document/d/19ZiExIc-vdbGrauuRUPKDI_o3sbVTeYMifc_hzvULGA/edit#heading=h.av0bqkw4ssvd)

4) Google Cloud Platform을 처음 사용해 보는 데 진입 장벽이 좀 있음
- 삽질 중인데 구글에 요청에서 가이드 문서나 reference가 될 수 있는 example을 받았으면 좋겠음

5) Tensorflow 단위 테스트 + 테스트 코드를 짜는 좋은 방법이 있으면 가이드를 받고 싶음


