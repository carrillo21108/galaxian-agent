# a2c_galaxian.py
# ------------------------------------------------------------
# A2C desde cero para ALE/Galaxian-v5 (Gymnasium + ALE-Py)
# Compatible con wrappers personalizados (env_utils.make_galaxian_env)
# - Observación por defecto: (4, 84, 84) (canal-primero)
# - Robusto si llega en (84, 84, 4) (canal-último): auto-detecta y permuta
# - Checkpoints periódicos en 'checkpoint_dir' (útil para Google Drive en Colab)
# ------------------------------------------------------------

import os
import csv
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para guardar PNG
import matplotlib.pyplot as plt

from env_utils import make_galaxian_env  # <- usa tus custom wrappers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Red Actor-Crítico (backbone conv compartido)
# ============================================================
class A2CNet(nn.Module):
    """
    Red Actor-Crítico con backbone convolucional estilo Nature.
    Espera tensores en float normalizados [0,1] con formato (B, C, H, W).
    Si recibe (B, H, W, C), permuta automáticamente a (B, C, H, W).
    """
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.expected_c = c  # para validar/ajustar layout en forward

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # -> (B,32,20,20) aprox con 84x84
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (B,64,9,9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (B,64,7,7)
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.features(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B,C,H,W) uint8/float o (B,H,W,C) uint8/float
        Retorna:
          - logits de la política: (B, n_actions)
          - valores V(s): (B,)
        """
        if x.ndim != 4:
            raise ValueError(f"Se esperaba un tensor 4D, recibido x.ndim={x.ndim}")

        # Si viene en canal-último (B,H,W,C), permutamos a (B,C,H,W)
        if x.shape[1] != self.expected_c and x.shape[-1] == self.expected_c:
            x = x.permute(0, 3, 1, 2)

        x = x.float() / 255.0
        feat = self.features(x)
        logits = self.actor(feat)
        values = self.critic(feat).squeeze(-1)
        return logits, values


# ============================================================
# Política para play.py (interfaz: __call__(obs, info) -> action)
# ============================================================
class A2CPolicy:
    def __init__(self, net: 'A2CNet'):
        self.net = net.to(device)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, info: dict) -> int:
        """
        obs: (C,H,W) uint8 o (H,W,C) uint8
        Estrategia greedy (argmax) para evaluación/competencia.
        """
        if obs.ndim != 3:
            raise ValueError(f"Se esperaba obs 3D, recibido obs.ndim={obs.ndim}")

        obs_batch = np.expand_dims(obs, axis=0)  # (1, C,H,W) o (1, H,W,C)
        obs_t = torch.from_numpy(obs_batch).to(device)
        logits, _ = self.net(obs_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        return int(action)


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
    plt.title("Training Rewards - A2C")
    plt.legend()
    png_path = os.path.join(checkpoint_dir, f"rewards_ep{episode}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[LOG] Guardadas gráfica y CSV en: {png_path} / {csv_path}")


# ============================================================
# Entrenamiento A2C
# ============================================================
def train_a2c(
    checkpoint_dir: str,
    total_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 2.5e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    rollout_length: int = 5,
    save_interval: int = 50,
    plot_interval: int = 50,
    max_steps_per_episode: int | None = None,
    seed: int = 123,
    gae_lambda: float = 0.95,
):
    """
    Entrena un A2C minimalista para Galaxian con rollouts cortos (n-steps) y GAE(λ).
    Guarda checkpoints periódicos en `checkpoint_dir` (ideal Drive en Colab).
    Guarda gráficas de recompensa y CSV cada `plot_interval` episodios.
    Retorna la red A2C entrenada.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = make_galaxian_env(seed=seed, render_mode=None)

    # Detectar layout y shape de entrada
    obs, _ = env.reset()
    if obs.ndim != 3:
        env.close()
        raise ValueError(f"Observación inesperada: ndim={obs.ndim}, se esperaba 3")

    if obs.shape[0] in (1, 3, 4):  # canal-primero típico de CustomFrameStack
        input_shape = (obs.shape[0], obs.shape[1], obs.shape[2])
    else:                           # canal-último
        input_shape = (obs.shape[2], obs.shape[0], obs.shape[1])

    n_actions = env.action_space.n

    net = A2CNet(input_shape, n_actions).to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, eps=1e-5)

    episode_idx = 0
    rewards_log: List[float] = []

    while episode_idx < total_episodes:
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps_in_ep = 0

        while not done:
            # Trajectoria corta (rollout)
            log_probs = []
            values = []
            rewards = []
            dones = []
            entropies = []

            for _ in range(rollout_length):
                if done:
                    break

                ob = np.expand_dims(obs, axis=0)             # (1, C,H,W) o (1, H,W,C)
                ob_t = torch.from_numpy(ob).to(device)

                logits, value = net(ob_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                action = dist.sample()
                entropy = dist.entropy().mean()

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                log_probs.append(dist.log_prob(action).squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
                dones.append(torch.tensor(float(done), device=device))
                entropies.append(entropy)

                obs = next_obs
                ep_reward += float(reward)
                steps_in_ep += 1

                if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                    done = True
                    break

            # Bootstrap del valor para el último estado
            if done:
                next_value = torch.zeros(1, device=device)
            else:
                nb = np.expand_dims(obs, axis=0)
                nb_t = torch.from_numpy(nb).to(device)
                _, nv = net(nb_t)
                next_value = nv.detach()

            # GAE(λ) + retornos
            returns = []
            gae = 0
            # Usamos next_value como V(s_{t+T})
            v_next = next_value
            for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
                v_next = v_next * (1.0 - d)  # si done==1, no bootstrap
                delta = r + gamma * v_next - v
                gae = delta + gamma * gae_lambda * (1.0 - d) * gae
                v_next = v
                returns.insert(0, gae + v)

            returns = torch.stack(returns)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies) if len(entropies) > 0 else torch.tensor(0.0, device=device)

            advantage = returns - values_t

            policy_loss = -(log_probs_t * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            entropy_loss = entropies_t.mean() if entropies_t.ndim > 0 else entropies_t

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            if done:
                break  # salir al cierre del episodio

        episode_idx += 1
        rewards_log.append(ep_reward)
        print(f"[A2C] Episodio {episode_idx}/{total_episodes} | Recompensa: {ep_reward:.1f}")

        # Guardar checkpoint cada N episodios
        if episode_idx % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"a2c_galaxian_ep{episode_idx}.pth")
            torch.save({
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode_idx,
                "rewards": rewards_log,
                "input_shape": input_shape,
                "n_actions": n_actions,
            }, ckpt_path)
            print(f"[A2C] Checkpoint guardado en: {ckpt_path}")

        # Guardar gráfica + CSV cada N episodios
        if episode_idx % plot_interval == 0:
            _save_plot_and_csv(checkpoint_dir, rewards_log, episode_idx)

    env.close()

    # Guardar pesos finales para inferencia
    final_path = os.path.join(checkpoint_dir, "a2c_galaxian_final.pth")
    torch.save(net.state_dict(), final_path)
    print(f"[A2C] Modelo final guardado en: {final_path}")

    # Guardar gráfica final
    _save_plot_and_csv(checkpoint_dir, rewards_log, total_episodes)

    return net
