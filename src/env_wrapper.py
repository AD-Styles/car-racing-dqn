# src/env_wrapper.py
import gymnasium as gym
import numpy as np
import cv2

class CarRacingActionWrapper(gym.ActionWrapper):
    """
    연속적인 조작을 4가지 이산적인 행동(Discrete Action)으로 매핑.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        # [steering, gas, brake]
        self.action_mapping = {
            0: np.array([0.0, 0.0, 0.0]),  # 정지/관성 주행
            1: np.array([-1.0, 0.0, 0.0]), # 좌회전
            2: np.array([1.0, 0.0, 0.0]),  # 우회전
            3: np.array([0.0, 0.5, 0.0]),  # 직진 (가속)
        }

    def action(self, action):
        return self.action_mapping[action]

class CarRacingObservationWrapper(gym.ObservationWrapper):
    """
    RGB 화면을 84x84 흑백 이미지로 변환하고 정규화.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        # 하단의 대시보드 부분을 자르고 흑백 변환
        gray = cv2.cvtColor(obs[:84, :, :], cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 0~1 사이로 정규화
        return np.array(resized, dtype=np.float32) / 255.0

def make_env(render_mode=None):
    """
    전처리와 Frame Stacking이 모두 적용된 최종 환경을 반환.
    """
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    env = CarRacingActionWrapper(env)
    env = CarRacingObservationWrapper(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env
