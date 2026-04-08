# 🏎️ Autonomous Car-Racing with Deep Q-Network <br>(CNN-DQN 기반 Car-Racing 자율주행 에이전트 구축 및 웹 데모 배포<br> 프로젝트)

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

| 구분&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 세부 내용 |
| :--- | :--- |
| **연속적&nbsp;상태&nbsp;공간&nbsp;제어** | 단순한 좌표나 속도 값이 아닌, 96x96 RGB 픽셀 화면 자체를 상태(State)로 인식하고 처리하는 Vision 기반 강화학습 모델 설계 |
| **이산&nbsp;행동&nbsp;매핑** | Steering(조향), Gas(가속), Brake(감속)의 연속적 조작을 [좌회전, 직진, 우회전, 브레이크] 4가지 이산(Discrete) 행동으로 변환하여 학습 효율 극대화 |
| **안정적&nbsp;학습&nbsp;환경&nbsp;구축** | 시계열 데이터의 상관관계와 Q-value 발산 문제를 해결하기 위해 Experience Replay 및 Target Network 기법 적용 |

---

## 📁 프로젝트 구조 (Project Structure)

```text
📁car-racing-dqn/
├──📁 src
│   ├──📄 agent.py   # 이미지를 분석하는 CNN 신경망을 정의하고, Epsilon-Greedy 기반의 행동 선택과 미니 배치를 통한 Q-value 업데이트
│   └──📄app.py      # 환경 전처리, 학습 루프 통합 및 Hugging Face Space 배포 스크립트
├──📄 .gitignore
├──📄 LICENSE
├──📄 README.md
├──📄 requirements.txt
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

| 핵심 개념 | 비유 (Analogy)&nbsp;&nbsp;&nbsp;&nbsp; | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Epsilon-Greedy<br> 탐험** | **맛집 탐방** | 무조건 아는 맛집만 가는 것(가치 활용)이 아니라, 가끔은 새로운 식당(무작위 탐험)을 시도하여 더<br> 나은 식당을 찾는 과정 |
| **Experience Replay** | **랜덤 오답 노트<br> 복습** | 직전에 푼 문제만 이어서 풀면 편향이 생기므로, 과거의 경험들을 버퍼에 모아두고 무작위로 섞어서<br> 다시 복습하며 학습 안정성을 높이는 기법 |
| **Target Network<br> 분리** | **움직이는 과녁<br> 멈추기** | 정답(Target)이 계속 변하면 활을 맞추기 어려우므로, 일정 주기마다 과녁의 위치를 고정해두고 영점을 조절하는 방식 |

---


# 🏎️ 최종 결과물 - Huggingface-Space
👉 **[Play CarRacing Agent Live] ([https://huggingface.co/spaces/AD-Styles/CarRacing_Random_vs_DQN])**

## 🥊 에이전트 성능 비교 (Random vs Trained DQN)

본 프로젝트의 웹 데모에서는 에이전트의 학습 성과를 직관적으로 증명하기 위해, 아무런 학습을 거치지 않은 **랜덤 에이전트**와 학습이 완료된 **DQN 에이전트**의 주행 성능을 실시간으로 비교하였습니다.

| 비&#8288;교&#8288;&nbsp;&#8288;대&#8288;상 | 행동 결정 방식 (Action Selection) | 주행 결과 및 특징 |
| :--- | :--- | :--- |
| **랜&#8288;덤&#8288;&nbsp;&#8288;에&#8288;이&#8288;전&#8288;트**<br>**(Baseline)** | 현재 화면 상태(State)와 무관하게 4가지 이산 행동 중 하나를 무작위 확률로 선택. | 트랙의 궤적을 인지하지 못하고 무작위로 조향하여 빠르게 코스를 이탈 및 페널티 누적 |
| **D&#8288;Q&#8288;N&#8288;&nbsp;&#8288;에&#8288;이&#8288;전&#8288;트**<br>**(Trained)** | 누적된 프레임을 CNN으로 분석하여, 현재 상태에서 기대 보상(Q-Value)이 가장 높은 최적의 행동 선택. | 트랙의 곡률을 스스로 파악하고 적절한 타이밍에 감속 및 방향 전환을 수행하여 높은 보상 획득 |

---

## 💡 회고록 (Retrospective)

&emsp;이번 CarRacing 프로젝트를 통해 '날것의 이미지(Raw Pixel) 데이터를 어떻게 에이전트가 이해할 수 있는 상태(State)로 변환할 것인가?' 에 대한 깊은 고민을 할 수 있었습니다. 프로젝트 초기, 에이전트에게 주어진 환경은 오직 96x96 픽셀의 정지된 이미지 한 장이었습니다. 가장 큰 난관은 정지 화면만으로는 자동차가 앞으로 달리고 있는지, 뒤로 미끄러지는지 그 '동적 상태'를 파악할 수 없다는 점이었습니다. 이를 해결하기 위해 이미지를 흑백(Grayscale)으로 변환하고 84x84로 축소하여 연산량을 최적화하는 한편, 최근 4개의 프레임을 하나로 겹치는 'Frame Stacking' 기법을 파이프라인에 도입했습니다. 단순한 전처리를 넘어 데이터에 '시간적 맥락(Temporal Context)'을 부여한 이 결정은, 에이전트가 스스로 속도와 방향을 인지하고 부드럽게 코너를 도는 결과로 이어졌습니다.
<br>&emsp;또 다른 에로사항으로는, 실제 자동차처럼 조향(Steering), 가속(Gas), 감속(Brake)을 연속적인(Continuous) 값으로 제어하려고 하니 초기 탐색 공간이 지나치게 방대해져 에이전트가 갈피를 잡지 못했습니다. 이를 해결하기위해, 이 복잡한 조작을 '좌회전, 직진, 우회전, 브레이크'라는 4가지의 명확한 **이산 행동(Discrete Action)**으로 재정의했습니다. 완벽하고 세밀한 시뮬레이션 환경을 고집하기보다, 주어진 컴퓨팅 자원과 목표 안에서 AI가 가장 빠르고 정확하게 최적해를 찾아갈 수 있도록 문제 자체를 단순화하고 재설계하는 접근법을 도입했습니다.
<br>&emsp;이론으로 완벽해 보이던 DQN 모델은 실제 학습 과정에서 값이 발산(Divergence)하며 트랙을 사정없이 벗어나는 불안정성을 보였습니다. 이 문제를 잡기 위해 Experience Replay 버퍼에 모인 과거의 경험들을 무작위로 추출하여 데이터 간의 상관관계를 끊어내고 학습의 편향을 줄였습니다. 또한, Target Network의 업데이트 주기를 조절하여 정답(과녁)이 요동치는 현상을 통제했습니다. 나아가 에이전트가 익숙한 길만 고집하지 않고 새로운 경로를 탐색하도록 Epsilon-Greedy의 감소율(Decay rate)을 끈질기게 미세 조정했습니다. 수많은 실패 끝에 안정적인 우상향 학습 곡선을 그려냈을 때, 딥러닝 모델은 단순히 코드로 완성되는 것이 아니라, 데이터의 피드백을 읽고 끝없이 영점을 조절해 나가는 집요한 엔지니어링의 결과물 이라는것을 다시 한번 깨달았습니다.
<br>&emsp;최종적으로 에이전트가 트랙을 스스로 파악하고 부드럽게 코너링을 해내는 모습을 보았을 때, 이론으로만 배우던 딥러닝과 강화학습의 시너지를 온전히 실감할 수 있었습니다. 강화학습 특유의 까다로운 환경 속에서, 원시 데이터 전처리부터 CNN 아키텍처 설계, 그리고 모델 배포(Hugging Face Space)까지 전체 파이프라인을 직접 구축하고 수렴시켜 낸 이 경험은 제게 든든한 무기가 되었습니다. 이번 프로젝트에서 얻은 시스템 최적화 경험과 트러블슈팅 역량을 바탕으로, 앞으로 실무에서 마주할 복잡한 데이터 난제 앞에서도 가장 직관적이고 견고한 AI 솔루션을 설계하는 전문가로 성장해 나가겠다는 다짐을 하게된 소중한 프로젝트 였습니다.
