# app.py
import gradio as gr
import torch
import numpy as np
import imageio
from src.env_wrapper import make_env
from src.agent import DQN

def play_and_record():
    """
    학습된 에이전트를 1 에피소드 동안 실행하고 비디오 파일로 저장합니다.
    """
    # 렌더링 모드로 환경 설정
    env = make_env(render_mode="rgb_array")
    device = torch.device("cpu") # 웹 배포 환경을 고려하여 CPU 사용
    
    # 모델 구조 초기화 및 가중치 로드
    model = DQN(action_dim=4).to(device)
    # 로컬 경로 또는 허깅페이스 리포지토리의 파일 경로를 지정하세요.
    model.load_state_dict(torch.load('models/dqn_carracing_final.pth', map_location=device))
    model.eval() # 평가 모드

    state, _ = env.reset()
    done = False
    frames = []
    
    # 1 에피소드 주행 (최대 600 프레임 제한)
    while not done and len(frames) < 600:
        # 화면 캡처 저장
        frames.append(env.render())
        
        # 텐서 변환 및 추론 (탐험 없음)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()
            
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    env.close()
    
    # 비디오 파일 생성
    video_path = "car_racing_demo.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    
    return video_path

# Gradio 인터페이스 구성
demo = gr.Interface(
    fn=play_and_record,
    inputs=None,
    outputs=gr.Video(label="Autonomous Driving Demo"),
    title="🏎️ CNN-DQN CarRacing Agent",
    description="버튼을 누르면 에이전트가 트랙을 스스로 파악하여 주행하는 영상을 렌더링하여 보여줍니다. (약 10~20초 소요)",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
