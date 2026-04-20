# Titanic Survival Analysis

Kaggle Titanic 대회 데이터를 활용한 생존자 예측 머신러닝 프로젝트입니다.

## 프로젝트 구조

```
titanic-survival-analysis/
├── data/
│   ├── train.csv              # Kaggle 원본 학습 데이터 (891행)
│   ├── test.csv               # Kaggle 원본 테스트 데이터 (418행)
│   ├── X_train.csv            # 전처리 완료된 학습 피처
│   ├── X_test.csv             # 전처리 완료된 테스트 피처
│   └── y_train.csv            # 학습 타겟 레이블
├── notebooks/
│   ├── 01_EDA.ipynb           # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb # 데이터 전처리 & 피처 엔지니어링
│   └── 03_modeling.ipynb      # 모델 학습 & 하이퍼파라미터 튜닝
└── outputs/
    ├── eda_overview.png        # EDA 시각화 결과
    └── submission_lr_tuned.csv # 최종 제출 파일
```

## 분석 흐름

### 1. 탐색적 데이터 분석 (01_EDA.ipynb)
- 결측치 현황 파악 (Age 177개, Cabin 687개, Embarked 2개)
- 전체 생존율: **38%**
- 주요 인사이트
  - 성별: 여성 생존율이 남성보다 압도적으로 높음
  - 객실등급: 1등급일수록 생존율 높음
  - 나이: 어린이(12세 이하) 생존율 상대적으로 높음
  - 호칭(Title) 추출 및 생존율 분석

### 2. 전처리 & 피처 엔지니어링 (02_preprocessing.ipynb)
- **결측치 처리**
  - Age: 호칭(Title)별 중앙값으로 대체
  - Embarked: 최빈값으로 대체
  - Fare: 중앙값으로 대체
- **파생 변수 생성**
  - `FamilySize` = SibSp + Parch + 1
  - `IsAlone`: 혼자 탑승 여부
  - `HasCabin`: 객실 번호 보유 여부
  - `Woman_1class`, `Woman_2class`: 여성 × 등급 조합
  - `IsChild`: 12세 이하 여부
  - `Fare`: 이상치 클리핑 (상한 300)
- **인코딩**: Sex, Embarked, Title → LabelEncoder
- **최종 피처 12개**: Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone, HasCabin, Woman_1class, Woman_2class, IsChild

### 3. 모델 학습 (03_modeling.ipynb)
- **알고리즘**: Logistic Regression
- **튜닝 방법**: GridSearchCV (5-fold CV)
- **탐색 파라미터**: C, penalty (l1/l2), solver
- **최적 파라미터**: `C=10.0, penalty=l1, solver=liblinear`
- **최고 CV 정확도**: **81.60%**

## 사용 라이브러리

| 라이브러리 | 용도 |
|---|---|
| pandas | 데이터 처리 |
| matplotlib / seaborn | 시각화 |
| scikit-learn | 전처리 및 모델링 |

## 실행 방법

```bash
# 가상환경 활성화
venv\Scripts\activate

# 노트북 순서대로 실행
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

## 결과

Logistic Regression (튜닝 후) CV 정확도 **81.60%** 달성.
Kaggle 제출 점수: **0.78**
최종 예측 결과는 `outputs/submission_lr_tuned.csv`에 저장됩니다.

