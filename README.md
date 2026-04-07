# 🏎️ Autonomous Car Racing with Deep Q-Network<br> (CNN-DQN 기반 `CarRacing` 자율주행 에이전트 구축 및 웹 데모 배포 프로젝트)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)

---

## 📌 프로젝트 요약 (Project Overview)
**CNN-DQN 기반 `CarRacing` 자율주행 에이전트**

본 프로젝트는 강화학습 알고리즘인 **DQN(Deep Q-Network)**을 구현하여, OpenAI Gymnasium의 `CarRacing-v3` 환경에서 자동차가 트랙을 벗어나지 않고 자율 주행하도록 학습시키는 과정을 담은 프로젝트입니다. 연속적인 픽셀 이미지 데이터(주행 화면)를 CNN으로 직접 분석하고 최적의 주행 조작을 스스로 도출해내는 전체 파이프라인의 구축 및 배포 경험을 기록했습니다.

---

## 🎯 핵심 목표 (Motivation)

| 구분 | 세부 내용 |
| :--- | :--- |
| **연속적 상태 공간 제어** | 단순한 좌표나 속도 값이 아닌, 96x96 RGB 픽셀 화면 자체를 상태(State)로 인식하고 처리하는 Vision 기반 강화학습 모델 설계. |
| **이산 행동 매핑** | Steering(조향), Gas(가속), Brake(감속)의 연속적 조작을 [좌회전, 직진, 우회전, 브레이크] 4가지 이산(Discrete) 행동으로 변환하여 학습 효율 극대화. |
| **안정적 학습 환경 구축** | 시계열 데이터의 상관관계와 Q-value 발산 문제를 해결하기 위해 Experience Replay 및 Target Network 기법 적용. |

---

## 📁 프로젝트 구조 (Project Structure)

```text
📁car-racing-dqn/
├──📁 src
├── agent.py    # 이미지를 분석하는 CNN 신경망을 정의하고, Epsilon-Greedy 기반의 행동 선택과 미니 배치를 통한 Q-value 업데이트
├── app.py      # 환경 전처리, 학습 루프 통합 및 Hugging Face Space 배포 스크립트
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🧠 핵심 파이프라인 (Core Pipeline)

에이전트가 화면을 보고 판단을 내리기까지의 과정.

| 데이터 파이프라인 | 상세 처리 내용 | 처리 모듈 |
| :--- | :--- | :--- |
| **Action Mapping** | 원래의 연속적인 조작(Steering, Gas, Brake)을 4가지의 이산적(Discrete) 행동으로 매핑하여 DQN의 출력으로 사용 | `app.py` |
| **Grayscale & Resizing** | 3채널 RGB 이미지를 1채널 흑백으로 변환하고, 84x84 크기로 축소 및 정규화 | `app.py` |
| **Frame Stacking** | 정지된 프레임에서 속도와 방향을 파악하기 위해 최근 4개의 프레임을 누적(`4x84x84`) | `app.py` |
| **CNN Architecture** | 3개의 Conv Layer를 거쳐 특징을 추출하고, 2개의 FC Layer를 통해 4가지 행동에 대한 Q-Value 산출 | `agent.py` |

---

## 📊 학습 개념의 직관적 해석 (Analogies)

이번 프로젝트에 사용된 강화학습의 핵심 개념들을 직관적인 비유 정리.

| 핵심 개념 | 비유 (Analogy) | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Epsilon-Greedy 탐험** | **맛집 탐방** | 무조건 아는 맛집만 가는 것(가치 활용)이 아니라, 가끔은 새로운 식당(무작위 탐험)을 시도하여 더 나은 식당을 찾는 과정 |
| **Experience Replay** | **랜덤 오답 노트 복습** | 직전에 푼 문제만 이어서 풀면 편향이 생기므로, 과거의 경험들을 버퍼에 모아두고 무작위로 섞어서 다시 복습하며 학습 안정성을 높이는 기법 |
| **Target Network 분리** | **움직이는 과녁 멈추기** | 정답(Target)이 계속 변하면 활을 맞추기 어려우므로, 일정 주기마다 과녁의 위치를 고정해두고 영점을 조절하는 방식 |

---

# 🏎️ 최종 결과물 - Huggingface-Space
👉 **[Play CarRacing Agent Live] ([https://huggingface.co/spaces/AD-Styles/CarRacing_Random_vs_DQN])**

---

## 💡 회고록 (Retrospective)

&emsp;이번 CarRacing 프로젝트를 통해 **'날것의 이미지(Raw Pixel) 데이터를 어떻게 에이전트가 이해할 수 있는 상태(State)로 변환할 것인가?'**에 대한 깊은 고민을 할 수 있었습니다.
<br>&emsp;특히, 정지된 한 장의 이미지만으로는 자동차가 앞으로 가고 있는지 뒤로 가고 있는지 알 수 없다는 문제를 마주했을 때, **Frame Stacking(프레임 누적)** 기법을 통해 이미지 데이터에 '시간적 맥락'을 부여하여 해결했던 과정이 가장 기억에 남습니다. 
<br>&emsp;또한, 강화학습 특유의 불안정성을 경험하며 모델이 발산(Divergence)하는 현상을 겪기도 했습니다. 이를 해결하기 위해 Replay Buffer의 크기, Epsilon 감소율(Decay rate), Target Network 업데이트 주기 등 하이퍼파라미터가 모델 안정성에 미치는 영향을 직접 눈으로 확인하며 튜닝하는 값진 경험을 얻었습니다. 최종적으로 에이전트가 트랙을 스스로 파악하고 부드럽게 코너링을 해내는 모습을 보았을 때, 이론으로만 배우던 딥러닝과 강화학습의 시너지를 온전히 실감할 수 있었습니다.
