# src/train.py
import torch
import numpy as np
import os
from env_wrapper import make_env
from agent import DQNAgent

def train():
    # 1. 환경 및 에이전트 초기화
    env = make_env()
    agent = DQNAgent(action_dim=4, lr=1e-4, epsilon=1.0)
    
    # 하이퍼파라미터
    num_episodes = 500
    batch_size = 64
    target_update_freq = 10
    
    # 모델 저장 디렉토리 생성
    os.makedirs('../models', exist_ok=True)

    print("🚀 학습을 시작합니다...")
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # 1 에피소드 진행 (차가 트랙을 벗어나거나 완주할 때까지)
        while not done:
            # 행동 선택 및 환경 상호작용
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 버퍼에 경험 저장 및 학습
            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 너무 오래 걸리면 에피소드 강제 종료 (무한 루프 방지)
            if step_count > 1000:
                break
                
        # 타겟 네트워크 업데이트
        if episode % target_update_freq == 0:
            agent.update_target()
            
        # Epsilon 감소 (점점 탐험을 줄이고 학습된 지식 활용)
        agent.decay_epsilon()
        
        print(f"Episode: {episode} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
        
        # 50 에피소드마다 중간 모델 저장
        if episode % 50 == 0:
            torch.save(agent.q_net.state_dict(), f'../models/dqn_carracing_ep{episode}.pth')

    # 최종 모델 저장
    torch.save(agent.q_net.state_dict(), '../models/dqn_carracing_final.pth')
    print("✅ 학습 및 모델 저장이 완료되었습니다!")
    env.close()

if __name__ == "__main__":
    train()
