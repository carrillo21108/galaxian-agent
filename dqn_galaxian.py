# dqn_galaxian.py
# ------------------------------------------------------------
# DQN desde cero para ALE/Galaxian-v5 (Gymnasium + ALE-Py)
# Compatible con wrappers personalizados (env_utils.make_galaxian_env)
# - Observación por defecto: (4, 84, 84) (canal-primero)
# - Robusto si viene en (84, 84, 4) (canal-último): auto-detecta y permuta
# - Checkpoints periódicos en 'checkpoint_dir' (útil para Google Drive en Colab)
# ------------------------------------------------------------

import os
import csv
import random
from collections import deque
from typing import Tuple, Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para guardar PNG
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Red DQN (estilo Nature)
# ============================================================
class DQN(nn.Module):
    """
    Red convolucional para estimar Q(s,a).
    Espera tensores en float normalizados [0,1] con formato (B, C, H, W).
    Si recibe (B, H, W, C) permuta automáticamente a (B, C, H, W).
    """
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.expected_c = c  # Para validar/ajustar layout en forward

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # -> (B,32,20,20) aprox con 84x84
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (B,64,9,9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (B,64,7,7)
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Inferir tamaño del flatten de forma programática
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) uint8/float o (B,H,W,C) uint8/float
        Retorna Q(s,·): (B, n_actions)
        """
        if x.ndim != 4:
            raise ValueError(f"Se esperaba un tensor 4D, recibido x.ndim={x.ndim}")

        # Si viene en canal-último (B,H,W,C), permutamos a (B,C,H,W)
        if x.shape[1] != self.expected_c and x.shape[-1] == self.expected_c:
            x = x.permute(0, 3, 1, 2)

        # Asegurar float y normalizar
        x = x.float() / 255.0
        x = self.features(x)
        return self.head(x)


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        # Guardamos en bruto (uint8 para ahorrar memoria)
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        # Convertimos tipos; estados se quedan en uint8 para pasar a torch y normalizar allí
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Política para play.py (interfaz: __call__(obs, info) -> action)
# ============================================================
class DQNPolicy:
    def __init__(self, q_net: 'DQN'):
        self.q_net = q_net.to(device)
        self.q_net.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, info: dict) -> int:
        """
        obs: (C,H,W) uint8 o (H,W,C) uint8
        """
        if obs.ndim != 3:
            raise ValueError(f"Se esperaba obs 3D, recibido obs.ndim={obs.ndim}")

        # Meter batch
        obs_batch = np.expand_dims(obs, axis=0)  # (1, C,H,W) o (1, H,W,C)
        obs_t = torch.from_numpy(obs_batch).to(device)
        q_vals = self.q_net(obs_t)
        return int(torch.argmax(q_vals, dim=1).item())


# ============================================================
# Utilidades de logging
# ============================================================
def _moving_average(x: List[float], k: int = 100):
    """Calcula la media móvil de una lista de valores."""
    if len(x) == 0:
        return []
    out = []
    s = 0.0
    q = []
    for i, v in enumerate(x):
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def _save_plot_and_csv(checkpoint_dir: str, rewards: List[float], episode: int):
    """Guarda gráfica de recompensas (PNG) y registro CSV."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Guardar CSV
    csv_path = os.path.join(checkpoint_dir, "rewards_log.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "reward"])
        w.writerow([episode, rewards[-1]])

    # Guardar gráfica PNG
    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards, label="Reward")
    ma = _moving_average(rewards, k=100)
    if len(ma) > 0:
        plt.plot(ma, label="Moving Avg (100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards - DQN")
    plt.legend()
    png_path = os.path.join(checkpoint_dir, f"rewards_ep{episode}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[LOG] Guardadas gráfica y CSV en: {png_path} / {csv_path}")


# ============================================================
# Entrenamiento DQN
# ============================================================
def train_dqn(
    checkpoint_dir: str,
    total_episodes: int = 500,
    replay_size: int = 100_000,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-4,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.1,
    epsilon_decay_episodes: int = 300,
    target_update_interval: int = 1_000,  # en pasos
    train_start: int = 10_000,            # tamaño mínimo de buffer para empezar a entrenar
    save_interval: int = 50,              # guardar checkpoints cada N episodios
    plot_interval: int = 50,              # guardar gráfica y CSV cada N episodios
    max_steps_per_episode: int | None = None,
    seed: int = 42,
):
    """
    Entrena un DQN minimalista para Galaxian.
    Guarda checkpoints periódicos en `checkpoint_dir` (ideal en /content/drive/... en Colab).
    Guarda gráficas de recompensa y CSV cada `plot_interval` episodios.
    Retorna la red Q entrenada (q_net).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Entorno con wrappers personalizados (sin depender de gymnasium.wrappers)
    env = make_galaxian_env(seed=seed, render_mode=None)

    # Obtenemos shape de la observación y detectamos layout
    obs, _ = env.reset()
    if obs.ndim != 3:
        env.close()
        raise ValueError(f"Observación inesperada: ndim={obs.ndim}, se esperaba 3")

    # Auto-detectar (C,H,W) vs (H,W,C) y construir input_shape=(C,H,W)
    if obs.shape[0] in (1, 3, 4):  # canal-primero típico de CustomFrameStack
        input_shape = (obs.shape[0], obs.shape[1], obs.shape[2])
    else:                           # canal-último
        input_shape = (obs.shape[2], obs.shape[0], obs.shape[1])

    n_actions = env.action_space.n

    q_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_size)

    global_step = 0

    def epsilon_by_episode(ep: int) -> float:
        if ep >= epsilon_decay_episodes:
            return end_epsilon
        frac = ep / float(epsilon_decay_episodes)
        return start_epsilon + frac * (end_epsilon - start_epsilon)

    # (Opcional) logging sencillo
    rewards_log: List[float] = []

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        eps = epsilon_by_episode(episode)
        steps_in_ep = 0

        while not done:
            global_step += 1
            steps_in_ep += 1

            # Epsilon-greedy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    ob = np.expand_dims(obs, axis=0)  # (1, C,H,W) o (1, H,W,C)
                    ob_t = torch.from_numpy(ob).to(device)
                    q_vals = q_net(ob_t)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            # Guardamos en buffer (mantener dtype uint8 en estados y next_estados)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # Entrenar cuando el buffer tenga suficientes transiciones
            if len(buffer) >= train_start:
                s, a, r, ns, d = buffer.sample(batch_size)

                # Convertir a tensores
                s_t = torch.from_numpy(s).to(device)
                ns_t = torch.from_numpy(ns).to(device)
                a_t = torch.from_numpy(a).long().to(device)
                r_t = torch.from_numpy(r).float().to(device)
                d_t = torch.from_numpy(d.astype(np.float32)).to(device)

                # Q(s,a)
                q_vals = q_net(s_t)                       # (B, n_actions)
                q_a = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)  # (B,)

                # y = r + gamma * max_a' Q_target(ns, a') * (1 - done)
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1)[0]
                    target = r_t + gamma * next_q * (1.0 - d_t)

                loss = nn.functional.mse_loss(q_a, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            # Actualizar target network cada cierto número de pasos
            if global_step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            # Límite duro de pasos por episodio (opcional)
            if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                break

        rewards_log.append(total_reward)
        print(f"[DQN] Episodio {episode}/{total_episodes} | Recompensa: {total_reward:.1f} | eps={eps:.3f} | buffer={len(buffer)}")

        # Guardar checkpoint en Google Drive cada N episodios
        if episode % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dqn_galaxian_ep{episode}.pth")
            torch.save({
                "q_net": q_net.state_dict(),
                "target_net": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "global_step": global_step,
                "rewards": rewards_log,
                "input_shape": input_shape,
                "n_actions": n_actions,
            }, ckpt_path)
            print(f"[DQN] Checkpoint guardado en: {ckpt_path}")

        # Guardar gráfica + CSV cada N episodios
        if episode % plot_interval == 0:
            _save_plot_and_csv(checkpoint_dir, rewards_log, episode)

    env.close()

    # Guardar pesos finales (solo la Q principal para inferencia)
    final_path = os.path.join(checkpoint_dir, "dqn_galaxian_final.pth")
    torch.save(q_net.state_dict(), final_path)
    print(f"[DQN] Modelo final guardado en: {final_path}")

    # Guardar gráfica final
    _save_plot_and_csv(checkpoint_dir, rewards_log, total_episodes)

    return q_net
