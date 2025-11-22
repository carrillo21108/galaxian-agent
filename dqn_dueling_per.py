# dqn_dueling_per.py
# ------------------------------------------------------------
# Dueling Double DQN con Prioritized Experience Replay (PER)
# - Compatible con env_utils.make_galaxian_env (custom wrappers)
# - Robusto a (C,H,W) o (H,W,C)
# - Checkpoints periódicos en Google Drive
# - Gráficas de recompensa y media móvil (PNG) + log CSV
# ------------------------------------------------------------

import os
import csv
import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para guardar PNG
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
#  Segment Trees para PER
# ===========================
class SegmentTree:
    def __init__(self, capacity, fn):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, \
            "capacity debe ser potencia de 2"
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.fn = fn

    def update(self, idx, value):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.fn(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def reduce(self, start, end):
        res_left = None
        res_right = None
        start += self.capacity
        end += self.capacity
        while start <= end:
            if (start % 2) == 1:
                res_left = self.tree[start] if res_left is None else self.fn(res_left, self.tree[start])
                start += 1
            if (end % 2) == 0:
                res_right = self.tree[end] if res_right is None else self.fn(self.tree[end], res_right)
                end -= 1
            start //= 2
            end //= 2
        if res_left is None:
            return res_right
        if res_right is None:
            return res_left
        return self.fn(res_left, res_right)

    def __getitem__(self, idx):
        return self.tree[idx + self.capacity]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, fn=lambda a, b: a + b)

    def sum(self):
        return self.tree[1]

    def find_prefixsum_idx(self, prefixsum):
        """Devuelve el índice i tal que sum(0..i) >= prefixsum."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, fn=min)

    def min(self):
        return self.tree[1]


# =======================================
#  Prioritized Replay Buffer (proportional)
# =======================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        # capacity → potencia de 2 para segment tree
        pow2 = 1
        while pow2 < capacity:
            pow2 *= 2
        self.capacity = pow2
        self.alpha = alpha
        self.eps = eps

        self.pos = 0
        self.size = 0

        self.states = [None] * self.capacity
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = [None] * self.capacity
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)
        self.max_priority = 1.0

        # Inicializa árboles con prioridad mínima
        for i in range(self.capacity):
            self.sum_tree.update(i, 0.0)
            self.min_tree.update(i, float("inf"))

    def __len__(self):
        return self.size

    def add(self, s, a, r, ns, d):
        idx = self.pos
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = ns
        self.dones[idx] = d

        p = (self.max_priority + self.eps) ** self.alpha
        self.sum_tree.update(idx, p)
        self.min_tree.update(idx, p)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """Devuelve (indices, w, batch) con pesos de importancia."""
        out_idx = []
        out_s = []
        out_a = np.empty(batch_size, dtype=np.int64)
        out_r = np.empty(batch_size, dtype=np.float32)
        out_ns = []
        out_d = np.empty(batch_size, dtype=np.float32)

        total = self.sum_tree.sum()
        segment = total / batch_size
        min_prob = self.min_tree.min() / total
        max_w = (min_prob * self.size) ** (-beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.random() * (b - a) + a
            idx = self.sum_tree.find_prefixsum_idx(mass)
            out_idx.append(idx)
            out_s.append(self.states[idx])
            out_a[i] = self.actions[idx]
            out_r[i] = self.rewards[idx]
            out_ns.append(self.next_states[idx])
            out_d[i] = float(self.dones[idx])

        # pesos de importancia
        probs = np.array([self.sum_tree[idx] / total for idx in out_idx], dtype=np.float32)
        w = (probs * self.size) ** (-beta)
        w = w / max_w
        w = w.astype(np.float32)

        return np.array(out_idx), w, (np.array(out_s), out_a, out_r, np.array(out_ns), out_d)

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            p = float(p + self.eps)
            self.sum_tree.update(i, (p) ** self.alpha)
            self.min_tree.update(i, (p) ** self.alpha)
            self.max_priority = max(self.max_priority, p)


# ===========================
#  Red Dueling
# ===========================
class DuelingDQN(nn.Module):
    """
    Backbone conv + (stream Valor) + (stream Ventaja). Q = V + (A - mean(A))
    Espera tensores (B,C,H,W) float/uint8; si llega (B,H,W,C) permuta.
    """
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.expected_c = c

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, c, h, w)).shape[1]

        self.value = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"x.ndim={x.ndim}, esperado 4")
        if x.shape[1] != self.expected_c and x.shape[-1] == self.expected_c:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        feat = self.features(x)
        v = self.value(feat)                  # (B,1)
        a = self.advantage(feat)              # (B,A)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


# ===========================
#  Política para play.py
# ===========================
class DQNPolicy:
    def __init__(self, q_net: 'DuelingDQN'):
        self.q_net = q_net.to(device)
        self.q_net.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, info: dict) -> int:
        if obs.ndim != 3:
            raise ValueError("obs debe ser 3D")
        ob = np.expand_dims(obs, axis=0)
        ob_t = torch.from_numpy(ob).to(device)
        q = self.q_net(ob_t)
        return int(torch.argmax(q, dim=1).item())


# ===========================
#  Utilidades de logging
# ===========================
def _moving_average(x: List[float], k: int = 100):
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    # CSV
    csv_path = os.path.join(checkpoint_dir, "rewards_log.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "reward"])
        w.writerow([episode, rewards[-1]])

    # PNG
    plt.figure(figsize=(8,4.5))
    plt.plot(rewards, label="Reward")
    ma = _moving_average(rewards, k=100)
    if len(ma) > 0:
        plt.plot(ma, label="Moving Avg (100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.legend()
    png_path = os.path.join(checkpoint_dir, f"rewards_ep{episode}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"[LOG] Guardadas gráfica y CSV en: {png_path} / {csv_path}")


# ===========================
#  Entrenamiento
# ===========================
def train_dueling_double_dqn_per(
    checkpoint_dir: str,
    total_episodes: int = 50000,
    buffer_capacity: int = 100_000,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-4,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.1,
    epsilon_decay_episodes: int = 30000,
    target_update_interval: int = 1000,     # en pasos
    train_start: int = 10_000,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_anneal_episodes: int = 50000,
    per_eps: float = 1e-6,
    save_interval: int = 500,               # guardar modelo
    plot_interval: int = 200,               # guardar PNG/CSV
    max_steps_per_episode: int | None = None,
    seed: int = 42,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = make_galaxian_env(seed=seed, render_mode=None)

    # Detecta layout y fija input_shape=(C,H,W)
    obs, _ = env.reset()
    if obs.ndim != 3:
        env.close()
        raise ValueError("obs.ndim inesperado")
    if obs.shape[0] in (1,3,4):
        input_shape = (obs.shape[0], obs.shape[1], obs.shape[2])
    else:
        input_shape = (obs.shape[2], obs.shape[0], obs.shape[1])
    n_actions = env.action_space.n

    q_net = DuelingDQN(input_shape, n_actions).to(device)
    target_net = DuelingDQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha, eps=per_eps)

    rewards_log: List[float] = []
    global_step = 0

    def epsilon_by_episode(ep):
        if ep >= epsilon_decay_episodes:
            return end_epsilon
        frac = ep / float(epsilon_decay_episodes)
        return start_epsilon + frac * (end_epsilon - start_epsilon)

    def beta_by_episode(ep):
        # lineal de per_beta_start -> per_beta_end
        frac = min(1.0, ep / float(per_beta_anneal_episodes))
        return per_beta_start + frac * (per_beta_end - per_beta_start)

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps_in_ep = 0

        eps = epsilon_by_episode(episode)
        beta = beta_by_episode(episode)

        while not done:
            global_step += 1
            steps_in_ep += 1

            # ε-greedy sobre q_net
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    ob = np.expand_dims(obs, axis=0)
                    ob_t = torch.from_numpy(ob).to(device)
                    q_vals = q_net(ob_t)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # Entrenamiento (cuando hay suficientes muestras)
            if len(buffer) >= train_start:
                idxs, isw, batch = buffer.sample(batch_size, beta=beta)
                s, a, r, ns, d = batch

                s_t  = torch.from_numpy(s).to(device)
                ns_t = torch.from_numpy(ns).to(device)
                a_t  = torch.from_numpy(a).long().to(device)
                r_t  = torch.from_numpy(r).float().to(device)
                d_t  = torch.from_numpy(d).float().to(device)
                w_t  = torch.from_numpy(isw).float().to(device)

                # Q_online(s,a)
                q = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

                # Double DQN target:
                # a* = argmax_a Q_online(ns,a)
                with torch.no_grad():
                    q_online_ns = q_net(ns_t)
                    a_star = torch.argmax(q_online_ns, dim=1, keepdim=True)

                    q_target_ns = target_net(ns_t)
                    next_q = q_target_ns.gather(1, a_star).squeeze(1)

                    target = r_t + gamma * next_q * (1.0 - d_t)

                td_error = target - q
                loss = (w_t * td_error.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                # actualizar prioridades con |TD error|
                new_prios = td_error.detach().abs().cpu().numpy() + per_eps
                buffer.update_priorities(idxs, new_prios)

            # sync target
            if global_step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                break

        rewards_log.append(total_reward)
        print(f"[Dueling-DDQN+PER] Ep {episode}/{total_episodes} | R: {total_reward:.1f} | eps={eps:.3f} | beta={beta:.3f} | buf={len(buffer)}")

        # Guardar checkpoint
        if episode % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dueling_ddqn_per_ep{episode}.pth")
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
            print(f"[CKPT] Guardado: {ckpt_path}")

        # Guardar gráfica + CSV
        if episode % plot_interval == 0:
            _save_plot_and_csv(checkpoint_dir, rewards_log, episode)

    env.close()

    # Guardar modelo final
    final_path = os.path.join(checkpoint_dir, "dueling_ddqn_per_final.pth")
    torch.save(q_net.state_dict(), final_path)
    print(f"[DONE] Modelo final guardado en: {final_path}")

    # Gráfica final
    _save_plot_and_csv(checkpoint_dir, rewards_log, total_episodes)

    return q_net