"""DDPG actor, critic, replay memory and native optimizer settings for S10."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim=60):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))


class Critic(nn.Module):
    def __init__(self, state_dim=60, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


class OUNoise:
    """Ornstein-Uhlenbeck process. Reset at the start of every design."""

    def __init__(self, size=1, mu=0.0, theta=0.15, sigma=0.3, rng=None):
        self.size, self.mu, self.theta, self.sigma = size, mu, theta, sigma
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.state = np.full(self.size, self.mu, dtype=np.float64)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.size)
        self.state = self.state + dx
        return self.state.copy()


class Replay:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=int(capacity))

    def push(self, s, a, r, s2, d):
        self.buf.append((np.asarray(s, np.float32), np.asarray(a, np.float32),
                         float(r), np.asarray(s2, np.float32), float(d)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.stack(a), np.asarray(r, np.float32).reshape(-1, 1),
                np.stack(s2), np.asarray(d, np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buf)


class DDPGAgent:
    def __init__(self, state_dim, device, actor_lr=1e-4, critic_lr=1e-3,
                 weight_decay=0.0, gamma=0.99, tau=0.001, batch_size=64,
                 memory_capacity=100000, ou=(0.0, 0.15, 0.3), noise_rng=None):
        self.device = device
        self.state_dim = int(state_dim)
        self.gamma, self.tau, self.batch_size = float(gamma), float(tau), int(batch_size)

        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)
        self.hard_update()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr,
                                          weight_decay=weight_decay)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr,
                                           weight_decay=weight_decay)
        self.memory = Replay(memory_capacity)
        self.noise = OUNoise(1, *ou, rng=noise_rng)
        self.mse = nn.MSELoss()
        self.n_updates = 0

    # ------------------------------------------------------------------ utils
    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _soft(self, net, target):
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

    # ------------------------------------------------------------------ act
    def act(self, state, noise=True):
        s = torch.as_tensor(np.asarray(state, np.float32), device=self.device).view(1, -1)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(s).cpu().numpy().reshape(-1)
        self.actor.train()
        if noise:
            a = a + self.noise.sample()
        return np.clip(a, -1.0, 1.0)

    # ------------------------------------------------------------------ learn
    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).view(-1, 1)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        with torch.no_grad():
            q_next = self.critic_target(s2, self.actor_target(s2))
            y = r + self.gamma * (1.0 - d) * q_next
        q = self.critic(s, a)
        critic_loss = self.mse(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft(self.actor, self.actor_target)
        self._soft(self.critic, self.critic_target)
        self.n_updates += 1
        return {"critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "q_mean": float(q.mean().item()),
                "q_target_mean": float(y.mean().item())}

    # ------------------------------------------------------------------ io
    def state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict()}

    def load_state_dicts(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic_target.load_state_dict(sd["critic_target"])
