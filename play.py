# play.py

import argparse
import datetime as dt
from pathlib import Path
from typing import Protocol, Optional, Tuple

import ale_py
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

from env_utils import make_galaxian_env
from dqn_galaxian import DQN, DQNPolicy
from dqn_dueling_per import DuelingDQN, DQNPolicy as DuelingDQNPolicy
from a2c_galaxian import A2CNet, A2CPolicy


class Policy(Protocol):
    def __call__(self, obs: np.ndarray, info: dict) -> int: ...


def _timestamp():
    return dt.datetime.now().strftime("%Y%m%d%H%M")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_dqn_policy(model_path: str) -> Policy:
    env = make_galaxian_env(seed=0, render_mode=None)
    # Forzamos el input_shape esperado por los custom wrappers:
    input_shape = (4, 84, 84)
    n_actions = env.action_space.n
    env.close()

    q_net = DQN(input_shape, n_actions)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "q_net" in state_dict:
        q_net.load_state_dict(state_dict["q_net"])
    else:
        q_net.load_state_dict(state_dict)
    return DQNPolicy(q_net)


def load_dueling_dqn_policy(model_path: str) -> Policy:
    env = make_galaxian_env(seed=0, render_mode=None)
    # Forzamos el input_shape esperado por los custom wrappers:
    input_shape = (4, 84, 84)
    n_actions = env.action_space.n
    env.close()

    q_net = DuelingDQN(input_shape, n_actions)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "q_net" in state_dict:
        q_net.load_state_dict(state_dict["q_net"])
    else:
        q_net.load_state_dict(state_dict)
    return DuelingDQNPolicy(q_net)


def load_a2c_policy(model_path: str) -> Policy:
    env = make_galaxian_env(seed=0, render_mode=None)
    # Forzamos el input_shape esperado por los custom wrappers:
    input_shape = (4, 84, 84)
    n_actions = env.action_space.n
    env.close()

    net = A2CNet(input_shape, n_actions)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "net" in state_dict:
        net.load_state_dict(state_dict["net"])
    else:
        net.load_state_dict(state_dict)
    return A2CPolicy(net)


def record_episode(
    policy: Policy,
    *,
    student_email: str,
    output_dir: str | Path = "videos",
    fps: int = 30,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, float]:
    """
    Ejecuta un episodio con la política dada sobre ALE/Galaxian-v5 preprocesado
    y genera un video MP4 con nombre:
        <correo>_<timestamp>_<score>.mp4
    
    Si seed es None, cada episodio será diferente (aleatorio).
    Si seed es un número, el episodio será reproducible.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    # Entorno con render para video
    env = make_galaxian_env(seed=seed, render_mode="rgb_array")
    obs, info = env.reset()
    frames = []

    # Primer frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if max_steps is not None and steps >= max_steps:
            break

    score_int = int(total_reward)
    filename = f"{student_email}_{_timestamp()}_{score_int}.mp4"
    video_path = output_dir / filename

    # Guardar video
    iio.imwrite(str(video_path), frames, fps=fps)
    env.close()

    print(f"Episodio terminado. Score: {total_reward:.1f}")
    print(f"Video guardado en: {video_path}")

    return video_path, total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Correo UVG para el nombre de archivo.")
    parser.add_argument("--algo", choices=["dqn", "dueling", "a2c"], required=True, help="Agente a usar.")
    parser.add_argument("--model-path", required=True, help="Ruta al .pth entrenado.")
    parser.add_argument("--output-dir", default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria (None para episodios aleatorios diferentes).")
    args = parser.parse_args()

    if args.algo == "dqn":
        policy = load_dqn_policy(args.model_path)
    elif args.algo == "dueling":
        policy = load_dueling_dqn_policy(args.model_path)
    else:
        policy = load_a2c_policy(args.model_path)

    record_episode(
        policy,
        student_email=args.email,
        output_dir=args.output_dir,
        fps=args.fps,
        max_steps=args.max_steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
