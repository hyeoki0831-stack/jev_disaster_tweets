JEV는 TypeSafe AI가 개발한 사전학습형 의사결정 모델로, 2026년 9월 15일 Early Access 형태로 공개된 첫 번째 System One Model이다. TypeSafe AI는 System One Model을 소프트웨어가 바로 활용할 수 있는 빠르고 구조화된 판단에 최적화된 모델로 설명하고 있다.

일반적인 GPT나 Claude 같은 생성형 LLM은 입력을 받은 뒤 토큰을 하나씩 생성하면서 자연어 답변을 만들어낸다. 반면 JEV는 긴 문장을 생성하는 것이 주목적이 아니라, 주어진 상태와 질문을 바탕으로 Boolean, Choice, Score 같은 구조화된 판단 결과와 확률을 반환하는 것에 초점이 맞춰져 있다.

기존의 Logistic Regression, LightGBM, BERT fine-tuning 같은 머신러닝 분류기와의 가장 큰 차이는, 새로운 분류 문제마다 별도의 학습이나 fine-tuning이 반드시 필요하지 않다는 점이다. 기존 지도학습 모델은 보통 정답 라벨이 있는 데이터를 준비하고 모델을 학습해야 하지만, JEV는 추론 시점에 자연어로 판단 기준과 선택지를 정의하여 zero-shot 방식으로 바로 사용할 수 있다.

또한 GPT나 Claude 같은 범용 생성형 LLM과 비교하면, JEV는 긴 텍스트 생성 대신 제한된 선택지에 대한 판단과 확률 출력에 집중하기 때문에 더 빠르고 저렴한 의사결정용 모델을 목표로 한다.

## 실험 목적

JEV는 별도의 task-specific fine-tuning 없이 자연어로 정의된 판단 기준에 따라 분류와 확률 기반 의사결정을 수행할 수 있는 사전학습 decision model이다.

기존의 머신러닝이나 딥러닝 기반 분류 문제에서는 일반적으로 정답 라벨이 포함된 학습 데이터를 준비하고, 해당 문제에 맞는 모델을 학습하거나 fine-tuning하는 과정이 필요하다.

이번 실험에서는 이러한 학습 과정 없이 JEV만으로 실제 분류 문제를 어느 정도 해결할 수 있는지 확인하고자 했다.

즉, 다음 질문에서 실험을 시작했다.

> **JEV가 별도의 학습 없이 기존 머신러닝·딥러닝 기반 분류 모델의 역할을 어느 정도 수행할 수 있을까?**

이를 확인하기 위해 Kaggle의 **Natural Language Processing with Disaster Tweets** 문제를 사용하였다.

이 대회의 목표는 주어진 트윗이 실제 재난 상황을 의미하는지 판단하여 다음과 같이 이진 분류하는 것이다.

- `1`: 실제 재난과 관련된 트윗
- `0`: 실제 재난이 아닌 트윗

일반적인 접근에서는 TF-IDF + Logistic Regression, BERT, RoBERTa 등의 모델을 학습하여 문제를 해결할 수 있다.

하지만 이번 실험에서는 이러한 별도의 분류 모델을 학습하지 않고, JEV에게 트윗과 판단 기준만 전달하였다.

예를 들어 JEV에게 다음과 같은 질문을 수행하도록 했다.

> 이 트윗이 실제 현실의 재난 또는 긴급 상황을 나타내는가?

JEV는 각 트윗에 대해 `True`에 해당하는 확률을 반환하며, 해당 확률을 기준으로 최종 `0/1` 예측값을 생성하였다.

## 실험 방식

실험은 두 가지 방식으로 진행하였다.

### 1. Pure Zero-shot JEV

JEV가 반환한 확률에 대해 기본 threshold인 `0.5`를 그대로 적용하였다.

```text
Tweet
  ↓
JEV
  ↓
P(real disaster)
  ↓
P >= 0.5 → 1
P < 0.5  → 0

2. JEV + Threshold Calibration
JEV 자체는 동일하게 zero-shot 방식으로 사용하되, Kaggle train 데이터 일부를 이용하여 F1 Score가 가장 높아지는 threshold를 탐색하였다.
즉 JEV의 weight를 학습하거나 fine-tuning한 것은 아니며, 최종 분류 기준만 데이터에 맞게 조정하였다.
```

### 결과

| Method | Kaggle Public F1 |
|---|---:|
| JEV Zero-shot (`threshold = 0.5`) | **0.82194** |
| JEV + Threshold Calibration | **0.83481** |

Threshold calibration을 적용했을 때 F1 Score가 약 0.01287 향상되었다.
특히 threshold를 별도로 조정하지 않은 순수 zero-shot 방식에서도 0.82194의 F1 Score를 기록했다는 점이 흥미로웠다.

### 해석
이번 실험에서는 JEV를 Kaggle Disaster Tweets 데이터로 별도로 학습시키지 않았다.
따라서 이번 결과는 JEV가 모든 머신러닝·딥러닝 모델을 대체할 수 있다는 의미는 아니다.
다만 텍스트의 의미를 기반으로 판단해야 하는 분류 문제에서는 별도의 task-specific 학습 없이도 기존 classifier와 유사한 역할을 수행할 가능성을 확인할 수 있었다.
특히 새로운 분류 기준이 자주 변경되거나, 충분한 라벨 데이터를 확보하기 어려운 상황에서는 JEV와 같은 사전학습 decision model이 하나의 대안이 될 수 있다고 생각한다.
