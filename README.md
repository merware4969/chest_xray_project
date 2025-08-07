
# DL_CHEST: 흉부 X-ray 기반 폐렴 이진 분류 프로젝트

본 프로젝트는 Kaggle Chest X-ray 데이터셋을 기반으로, 사전학습된 ResNet34 모델을 활용하여 폐렴 여부를 이진 분류하고,
Grad-CAM을 통해 모델의 의사결정 근거를 시각화하는 Streamlit 기반 의료 AI 프로젝트입니다.

---

## 📁 디렉터리 구조

```
DL_CHEST/
├── dlenv/                      # 가상환경
├── models/                    # 학습된 모델 저장 폴더
├── roc/                       # ROC 커브 시각화 결과 저장
├── saved_gradcam/             # Grad-CAM 결과 저장 폴더
├── test/                      # 테스트 이미지셋 (NORMAL / PNEUMONIA)
├── test_samples/              # Streamlit 테스트용 샘플 이미지
├── train_samples/             # Grad-CAM 테스트용 샘플 이미지
│
├── gen_roc.ipynb              # ROC 시각화 스크립트
├── streamlit_app.py           # Streamlit 앱 메인 실행 파일
├── fastai.ipynb               # fastai 기반 모델 성능 비교 실험 코드
├── resnet34_addval.ipynb      # ResNet34 학습 및 평가 코드 (val set 포함)
├── requirements.txt           # 패키지 목록
├── KoPubDotumMedium.ttf       # 한글 폰트
├── .gitignore                 # Git 무시 설정
├── README.md                  # 프로젝트 설명서
└── 딥러닝프로젝트_강충원.pdf     # 발표용 PDF 자료
```

---

## 📊 프로젝트 개요

- **목표**: 흉부 X-ray 이미지를 기반으로 폐렴 여부 이진 분류 및 Grad-CAM 시각화 적용
- **활용 모델**: 사전학습된 ResNet34 (ImageNet 기반)
- **데이터**: Kaggle 제공 JPEG 흉부 X-ray 데이터셋 (소아 1~5세 대상)
- **시각화**: Grad-CAM으로 예측 근거 영역 시각화
- **웹 앱**: Streamlit 기반 사용자 인터페이스 구현
- **개발 환경**: Google Colab (T4 GPU 환경) 기반 학습

---

## 🗂️ 데이터셋 정보

- **출처**: [Kaggle - Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **총 이미지 수**: 5,863장 (NORMAL / PNEUMONIA)
- **구성**:
  - 원본: train / val / test 폴더 구조
  - 본 프로젝트: train 데이터를 8:2로 분할하여 train / val 재구성
- **해상도**: 모든 이미지를 224×224로 리사이즈 후 입력
- **기타 정보**: 중국 광저우 여성·어린이병원 소아 환자 대상, 전문가의 수작업 라벨링 기반

---

## 🧠 모델 구성

- **Backbone**: ResNet34 (pretrained)
- **출력 구조**: `Linear(512 → 2)` → CrossEntropyLoss 사용
- **학습률 스케줄러**: `CyclicLR` 적용
  - `base_lr=1e-5`, `max_lr=1e-3`
  - `step_size_up = 2 에포크`
  - Adam 옵티마이저 + `cycle_momentum=False`
- **모델 비교 실험**: `fastai.ipynb` 파일을 통해 여러 모델을 1 epoch 학습 후 정확도 비교
  - 결과: `resnet34` > `efficientnet_b0` > `resnet18` > `densenet121` > `resnet50` → ResNet34 최종 채택

---

## 🔍 Grad-CAM 작동 방식

- 타깃 클래스의 gradient를 이용해 마지막 conv layer의 feature map에 가중치 부여
- 가중 평균된 activation map → 히트맵 생성 → 입력 이미지에 overlay
- 예측된 클래스 기준으로 시각적 근거 영역 표시

**한계점**
- 해상도 제한: 마지막 conv layer 기준이라 해상도가 낮음
- 클래스 특이성 부족: 특정 클래스 외 다중 질환 구분 어려움
- 의존성: 모델 아키텍처 및 훈련 품질에 따라 품질 달라짐

**개선 방향**
- 고해상도 Grad-CAM 변형 적용 고려
- 다층 시각화 (Score-CAM, Layer-CAM 등)
- 의료 전문가의 피드백 기반 개선

---

## 💡 느낀 점

- FastAI 기반 다양한 모델을 빠르게 실험하고 성능 비교해볼 수 있었음
- 복잡한 모델이 반드시 좋은 결과를 보장하지 않는다는 점 확인 (ResNet34 가장 우수)
- CyclicLR 스케줄러의 성능 향상 효과 체감
- Grad-CAM을 통해 예측의 해석 가능성을 높일 수 있었던 점이 가장 흥미로웠음

**아쉬운 점**
- JPEG 데이터셋이 아닌 DICOM 기반 의료 영상 전처리를 직접 적용해보지 못한 점
- TTA, 하이퍼컬럼 등의 기법 적용은 구조 및 셋업상 미적용

---

## ▶ 실행 방법

```bash
# 1. 가상환경 설정 후 패키지 설치
pip install -r requirements.txt

# 2. Streamlit 앱 실행
streamlit run streamlit_app.py
```

---

## 📚 참고 자료

- 논문: [Cell (2018)](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- Grad-CAM 원리: Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- PyTorch 공식 CyclicLR 문서: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html