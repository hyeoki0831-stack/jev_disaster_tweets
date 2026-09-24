JEV는 TypeSafe AI가 개발한 사전학습형 의사결정 모델로, 2026년 9월 15일 Early Access 형태로 공개된 첫 번째 System One Model이다. TypeSafe AI는 System One Model을 소프트웨어가 바로 활용할 수 있는 빠르고 구조화된 판단에 최적화된 모델로 설명하고 있다.

일반적인 GPT나 Claude 같은 생성형 LLM은 입력을 받은 뒤 토큰을 하나씩 생성하면서 자연어 답변을 만들어낸다. 반면 JEV는 긴 문장을 생성하는 것이 주목적이 아니라, 주어진 상태와 질문을 바탕으로 Boolean, Choice, Score 같은 구조화된 판단 결과와 확률을 반환하는 것에 초점이 맞춰져 있다.

기존의 Logistic Regression, LightGBM, BERT fine-tuning 같은 머신러닝 분류기와의 가장 큰 차이는, 새로운 분류 문제마다 별도의 학습이나 fine-tuning이 반드시 필요하지 않다는 점이다. 기존 지도학습 모델은 보통 정답 라벨이 있는 데이터를 준비하고 모델을 학습해야 하지만, JEV는 추론 시점에 자연어로 판단 기준과 선택지를 정의하여 zero-shot 방식으로 바로 사용할 수 있다.
또한 GPT나 Claude 같은 범용 생성형 LLM과 비교하면, JEV는 긴 텍스트 생성 대신 제한된 선택지에 대한 판단과 확률 출력에 집중하기 때문에 더 빠르고 저렴한 의사결정용 모델을 목표로 한다.
