# 🏎️ Autonomous Car Racing with Deep Q-Network (DQN)

> **CNN-DQN 기반 `CarRacing` 자율주행 에이전트 (프레임 스태킹, 경험 재생 적용)**

---

## 1. Project Structure
프로젝트는 재사용성과 유지보수를 고려하여 모듈화된 구조로 작성되었습니다.

```text
Car-Racing-DQN-Portfolio/
├── assets/                          # 시각화 그래프 및 플레이 영상 
├── models/                          # 학습된 모델 가중치 (dqn_carracing.pth)
├── notebooks/                       # 실험 및 검증용 주피터 파일
│   ├── 01_DQN_Agent_Basic.ipynb     # 행동선택 및 학습함수 구현
│   ├── 02_CarRacing_Env.ipynb       # CNN 확장 및 환경 테스트
│   └── 03_DQN_Training.ipynb        # 본격 학습 및 모델 저장
├── src/                             # 핵심 소스 코드
│   ├── env_wrapper.py               # 환경 전처리 (Frame Stacking 등)
│   ├── network.py                   # CNN 모델 아키텍처
│   ├── agent.py                     # DQNAgent 및 ReplayBuffer
│   └── train.py                     # 학습 파이프라인 루프
├── app.py                           # Hugging Face Space 배포 스크립트
├── requirements.txt                 # 의존성 패키지
└── README.md                        # 메인 포트폴리오 문서
```

## 2. Core Concepts: DQN
단순한 Q-Learning의 한계를 극복하기 위해 딥러닝을 결합한 DQN의 핵심 기술들을 에이전트에 적용했습니다.

| 핵심 개념 | 구현 및 적용 목적 |
| :--- | :--- |
| **Epsilon-Greedy 탐험** | 학습 초기에는 무작위 행동으로 환경을 탐험(Exploration)하고, 점진적으로 Epsilon 값을 감소시켜 학습된 신경망의 가치를 활용(Exploitation)하도록 균형을 맞춥니다. |
| **Experience Replay** | 에이전트의 경험 `(State, Action, Reward, Next State)`을 버퍼에 저장하고, 무작위로 미니 배치를 추출하여 학습합니다. 데이터 간의 상관관계를 끊어 학습 안정성을 높입니다. |
| **Target Network 분리** | Q값을 갱신할 때 타겟값이 계속 변동하는 불안정성을 막기 위해, 일정 주기마다 가중치가 동기화되는 별도의 Target Network를 유지합니다. |

---

## 3. Environment & Data Preprocessing
차량의 주행 화면(96x96 RGB 픽셀)을 에이전트가 효율적으로 학습할 수 있도록 전처리 파이프라인을 구축했습니다.

| 전처리 단계 | 상세 설명 및 목적 |
| :--- | :--- |
| **Action Mapping** | 원래의 연속적인 조작(Steering, Gas, Brake)을 [좌회전, 직진, 우회전, 브레이크] 4가지의 이산적(Discrete) 행동으로 매핑하여 DQN의 출력으로 사용합니다. |
| **Grayscale & Resizing** | 연산량을 줄이기 위해 3채널 RGB 이미지를 1채널 흑백으로 변환하고, 화면 크기를 84x84로 축소 및 정규화(0~1)합니다. |
| **Frame Stacking** | 정지된 1장의 이미지만으로는 차량의 속도와 이동 방향을 알 수 없으므로, 최근 4개의 프레임을 누적하여 하나의 State(`4x84x84`)로 모델에 입력합니다. |

---

## 4. Model Architecture (CNN)
시간적 맥락이 포함된 4채널 이미지 데이터를 처리하여 각 행동의 Q값을 예측하는 합성곱 신경망(CNN)을 설계했습니다.

| Layer | Type | Input Shape | Configuration | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Conv1** | 2D Convolution | `(4, 84, 84)` | 32 filters, Kernel 8x8, Stride 4 | ReLU |
| **Conv2** | 2D Convolution | `(32, 20, 20)` | 64 filters, Kernel 4x4, Stride 2 | ReLU |
| **Conv3** | 2D Convolution | `(64, 9, 9)` | 64 filters, Kernel 3x3, Stride 1 | ReLU |
| **Flatten** | - | `(64, 7, 7)` | 3136 features | - |
| **FC1** | Fully Connected | `3136` | 512 units | ReLU |
| **FC2** | Fully Connected | `512` | 4 units (Q-values) | Linear |

---

## 5. Training Results & Metrics
에이전트의 학습 성과를 나타내는 주요 지표입니다.

*(참고: `assets/` 폴더에 결과 이미지를 추가하고 아래 경로를 활성화하세요.)*

* **Reward Curve:** 에피소드가 진행됨에 따라 누적 보상이 어떻게 수렴하는지 보여줍니다.
  * `<img src="assets/reward_curve.png" width="600" alt="Reward Curve">`
* **Loss Curve:** Target Network와 예측 Q값 사이의 오차(Loss) 감소 추이를 나타냅니다.
  * `<img src="assets/loss_curve.png" width="600" alt="Loss Curve">`

---

## 6. Try it Live (Hugging Face Space)
학습이 완료된 모델은 브라우저에서 직접 실행해 볼 수 있도록 Hugging Face Space에 배포되었습니다.

👉 **[Play CarRacing Agent Live] (여기에 Hugging Face 스페이스 링크 삽입)**

* **Framework:** Gradio / PyTorch
* 모델이 실시간으로 프레임을 추론하여 주행하는 모습을 영상 스트리밍 형태로 확인 가능합니다.
