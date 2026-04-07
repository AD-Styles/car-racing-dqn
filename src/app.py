"""
CarRacing-v3: 랜덤 에이전트 vs DQN 에이전트 비교 데모
Hugging Face Spaces (Gradio)용 앱

기능:
1. 사전학습 모델 데모 (dqn_carracing.pth 로드)
2. 직접 학습시키기 (에피소드 수 선택 → 실시간 학습 → 결과 비교)
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import cv2
import random
import copy
import time
import tempfile
import os
from collections import deque

# ─────────────────────────────────────────────
# 1. 모델 & 환경 정의 (5-1 노트북과 동일)
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized.astype(np.float32) / 255.0


def discrete_to_continuous(action):
    action_map = {
        0: np.array([-0.5, 0.1, 0.0]),   # 좌회전 (가속을 0.1로 낮춤)
        1: np.array([0.0, 0.3, 0.0]),    # 직진 (가속을 0.3으로 낮춤)
        2: np.array([0.5, 0.1, 0.0]),    # 우회전 (가속을 0.1로 낮춤)
        3: np.array([0.0, 0.0, 0.8])     # 브레이크 (유지)
    }
    return action_map.get(action, np.array([0.0, 0.0, 0.0]))


class CarRacingWrapper:
    def __init__(self, render_mode=None):
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.frames = deque(maxlen=4)

    def reset(self):
        obs, _ = self.env.reset()
        p = preprocess_frame(obs)
        for _ in range(4):
            self.frames.append(p)
        return np.array(list(self.frames))

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(discrete_to_continuous(action))
        self.frames.append(preprocess_frame(obs))
        return np.array(list(self.frames)), r, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class DQN(nn.Module):
    def __init__(self, action_dim=4, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        with torch.no_grad():
            d = torch.zeros(1, input_channels, 84, 84)
            d = F.relu(self.conv1(d))
            d = F.relu(self.conv2(d))
            d = F.relu(self.conv3(d))
            self._cs = d.view(1, -1).size(1)
        self.fc1 = nn.Linear(self._cs, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────
# 2. ReplayBuffer & DQNAgent (5-1 노트북과 동일)
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, cap):
        self.buffer = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(np.array(a)),
                torch.FloatTensor(np.array(r)),
                torch.FloatTensor(np.array(ns)),
                torch.BoolTensor(np.array(d)))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, lr=0.0001, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 eps_decay=0.995, buf_size=10000, batch_size=32, target_update=1000):
        self.action_dim = 4
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.main_net = DQN(4).to(device)
        self.target_net = copy.deepcopy(self.main_net)
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buf_size)
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step_count = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.main_net(st).argmax(1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
        cq = self.main_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tq = r + self.gamma * self.target_net(ns).max(1)[0] * (~d).float()
        loss = F.smooth_l1_loss(cq, tq)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 10.0)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# ─────────────────────────────────────────────
# 3. 사전학습 모델 로드
# ─────────────────────────────────────────────

MODEL_PATH = "dqn_carracing.pth"

pretrained_model = DQN(4).to(device)
if os.path.exists(MODEL_PATH):
    try:
        pretrained_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
        pretrained_model.eval()
        MODEL_LOADED = True
        print(f"✅ 사전학습 모델 로드 완료: {MODEL_PATH}")
    except Exception as e:
        MODEL_LOADED = False
        print(f"❌ 모델 로드 실패: {e}")
else:
    MODEL_LOADED = False
    print(f"⚠️ 모델 파일({MODEL_PATH})이 없습니다.")


# ─────────────────────────────────────────────
# 4. 영상 녹화 함수
# ─────────────────────────────────────────────

def record_episode(model, use_model=True, max_steps=400):
    """에피소드 한 판을 녹화해서 mp4 경로와 총 보상을 반환"""
    env = CarRacingWrapper(render_mode="rgb_array")
    state = env.reset()
    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if use_model:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(st).argmax(1).item()
        else:
            action = random.randint(0, 3)

        state, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    env.close()

    if not frames:
        return None, 0.0

    h, w, _ = frames[0].shape
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 30, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp.name, total_reward


# ─────────────────────────────────────────────
# 5. DQN 학습 함수 (5-1 노트북의 train_dqn 기반)
# ─────────────────────────────────────────────

def train_dqn(num_episodes, progress=gr.Progress()):
    """DQN을 처음부터 학습하고, 학습 과정과 결과 영상을 반환"""
    num_episodes = int(num_episodes)
    max_steps = 400

    env = CarRacingWrapper(render_mode="rgb_array")
    agent = DQNAgent(eps_decay=0.99)
    episode_rewards = []
    episode_losses = []
    epsilons = []
    start_time = time.time()

    for episode in range(num_episodes):
        progress((episode + 1) / num_episodes,
                 desc=f"학습 중: 에피소드 {episode+1}/{num_episodes}")

        state = env.reset()
        ep_reward = 0
        ep_losses = []
        
        # [추가] 연속 감점(음수 보상)을 세는 카운터
        negative_reward_count = 0 

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # [추가] 트랙 이탈 방지 및 강제 종료 로직
            if reward < 0:
                negative_reward_count += 1
            else:
                negative_reward_count = 0 # 양수 보상을 받으면 카운터 초기화
            
            # 50 프레임(약 1.5초~2초) 연속으로 감점만 받았다면 길을 잃은 것으로 간주
            if negative_reward_count >= 50:
                done = True # 에피소드 강제 종료
                reward -= 20.0 # 트랙을 이탈한 것에 대한 강력한 페널티 부여

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            
            if loss is not None:
                ep_losses.append(loss)
            state = next_state
            ep_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0)
        epsilons.append(agent.epsilon)

    total_time = time.time() - start_time
    env.close()

    # 학습된 모델을 eval 모드로 전환
    agent.main_net.eval()

    # 학습 완료 후 결과 영상 녹화
    random_path, random_reward = record_episode(agent.main_net, use_model=False, max_steps=500)
    trained_path, trained_reward = record_episode(agent.main_net, use_model=True, max_steps=500)

    # 학습 로그 생성
    log_lines = []
    for i in range(len(episode_rewards)):
        if (i + 1) % 10 == 0 or i == 0 or i == len(episode_rewards) - 1:
            avg_r = np.mean(episode_rewards[max(0, i-9):i+1])
            log_lines.append(
                f"에피소드 {i+1:>4}/{num_episodes} | "
                f"보상: {episode_rewards[i]:>7.1f} | "
                f"최근10 평균: {avg_r:>7.1f} | "
                f"ε: {epsilons[i]:.3f}"
            )

    result_summary = (
        f"=== 학습 완료 ({num_episodes} 에피소드, {total_time/60:.1f}분 소요) ===\n"
        f"최종 평균 보상 (마지막 10): {np.mean(episode_rewards[-10:]):.1f}\n"
        f"최고 보상: {np.max(episode_rewards):.1f}\n"
        f"최저 보상: {np.min(episode_rewards):.1f}\n"
        f"\n--- 데모 결과 ---\n"
        f"🎲 랜덤: {random_reward:.1f}  vs  🧠 학습된 DQN: {trained_reward:.1f}  "
        f"({'DQN 승리! 🏆' if trained_reward > random_reward else '랜덤이 이김 😅' if random_reward > trained_reward else '무승부'})"
    )

    log_text = "\n".join(log_lines)

    return random_path, trained_path, result_summary, log_text


# ─────────────────────────────────────────────
# 6. 사전학습 모델 데모 핸들러
# ─────────────────────────────────────────────

def run_pretrained_demo():
    if not MODEL_LOADED:
        return None, None, "⚠️ 사전학습 모델 파일(dqn_carracing.pth)이 없습니다."

    random_path, random_reward = record_episode(pretrained_model, use_model=False, max_steps=400)
    trained_path, trained_reward = record_episode(pretrained_model, use_model=True, max_steps=400)

    info = (
        f"🎲 랜덤: {random_reward:.1f}  vs  🧠 사전학습 DQN: {trained_reward:.1f}  "
        f"({'DQN 승리! 🏆' if trained_reward > random_reward else '랜덤이 이김 😅' if random_reward > trained_reward else '무승부'})"
    )
    return random_path, trained_path, info


# ─────────────────────────────────────────────
# 7. Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="🏎️ CarRacing: Random vs DQN",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🏎️ CarRacing-v3 : 랜덤 vs DQN 에이전트
        강화학습(DQN)으로 학습한 자동차 에이전트와 랜덤 에이전트를 비교합니다.
        """
    )

    # ── 탭 1: 사전학습 모델 데모 ──
    with gr.Tab("📦 사전학습 모델 데모"):
        gr.Markdown(
            "### 미리 학습된 모델(dqn_carracing.pth)로 바로 비교\n"
            "이전에 학습하여 저장한 모델을 불러와 랜덤 에이전트와 비교합니다."
        )
        btn_pretrained = gr.Button("🏁 사전학습 모델 실행", variant="primary", size="lg")
        with gr.Row():
            vid_pre_r = gr.Video(label="🎲 랜덤 에이전트")
            vid_pre_t = gr.Video(label="🧠 사전학습 DQN 에이전트")
        txt_pre = gr.Textbox(label="비교 결과", interactive=False)

        btn_pretrained.click(
            fn=run_pretrained_demo,
            outputs=[vid_pre_r, vid_pre_t, txt_pre],
        )

    # ── 탭 2: 직접 학습시키기 ──
    with gr.Tab("🎓 직접 학습시키기"):
        gr.Markdown(
            "### DQN을 처음부터 학습시키고 결과를 확인\n"
            "에피소드 수를 선택하면 해당 횟수만큼 **실제로 학습**한 후 랜덤 에이전트와 비교합니다.\n"
            "에피소드가 많을수록 성능이 좋아지지만 학습 시간도 늘어납니다.\n\n"
            "⏱️ **예상 소요 시간** (CPU 기준): 50 에피소드 ~5분 / 100 에피소드 ~10분 / 300 에피소드 ~30분"
        )

        num_episodes = gr.Slider(
            10, 500, value=50, step=10,
            label="학습 에피소드 수",
            info="DQN 학습에 사용할 에피소드 수 (많을수록 성능 향상, 시간 증가)"
        )

        btn_train = gr.Button("🚀 학습 시작", variant="primary", size="lg")

        with gr.Row():
            vid_train_r = gr.Video(label="🎲 랜덤 에이전트")
            vid_train_t = gr.Video(label="🧠 학습된 DQN 에이전트")
        txt_train_result = gr.Textbox(label="학습 결과 요약", interactive=False)
        txt_train_log = gr.Textbox(label="학습 로그 (10 에피소드마다)", interactive=False, lines=10, max_lines=20)

        btn_train.click(
            fn=train_dqn,
            inputs=[num_episodes],
            outputs=[vid_train_r, vid_train_t, txt_train_result, txt_train_log],
        )

    # ── 하단 정보 ──
    gr.Markdown(
        """
        ---
        **사용 방법**
        1. **사전학습 모델 데모**: 미리 학습된 모델(dqn_carracing.pth)로 바로 결과를 확인합니다.
        2. **직접 학습시키기**: 에피소드 수를 선택하고 DQN을 처음부터 학습시킵니다.
           - 에피소드 수가 많을수록 더 잘 학습됩니다.
           - 학습 완료 후 랜덤 에이전트와 비교 영상을 자동으로 생성합니다.

        **모델 파일**: `dqn_carracing.pth` (이전 학습 노트북에서 저장한 파일)를 이 Space에 함께 업로드하세요.
        """
    )

# ─────────────────────────────────────────────
# 8. 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
