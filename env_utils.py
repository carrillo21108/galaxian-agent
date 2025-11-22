# env_utils
import gymnasium as gym
import ale_py
import numpy as np
import cv2
from collections import deque


class CustomAtariPreprocessing(gym.Wrapper):
    """
    Simplificado: convierte a escala de grises, reescala a (84, 84)
    y aplica frame_skip (por defecto 4).
    """

    def __init__(self, env, frame_skip=4, screen_size=84, grayscale_obs=True):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale_obs = grayscale_obs

        obs_shape = (screen_size, screen_size)
        if not grayscale_obs:
            obs_shape += (3,)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def process_frame(self, frame):
        """Convierte y reescala el frame a 84x84"""
        if self.grayscale_obs:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
            return frame
        else:
            frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
            return frame

    def step(self, action):
        """Ejecuta frame_skip pasos y devuelve el último frame procesado"""
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.frame_skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated |= term
            truncated |= trunc
            if terminated or truncated:
                break
        processed = self.process_frame(obs)
        return processed, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self.process_frame(obs)
        return processed, info


class CustomFrameStack(gym.Wrapper):
    """
    Apila los últimos N frames para captar la dinámica temporal.
    """

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low.min(), high=high.max(), dtype=env.observation_space.dtype, shape=(num_stack, *env.observation_space.shape)
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


def make_galaxian_env(seed: int | None = None, render_mode: str | None = None):
    """
    Crea el entorno ALE/Galaxian-v5 con preprocesamiento manual.
    - Reescalado a 84x84
    - Escala de grises
    - Frame skip = 4
    - Frame stack = 4
    """
    env = gym.make("ALE/Galaxian-v5", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)

    env = CustomAtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = CustomFrameStack(env, num_stack=4)

    return env
