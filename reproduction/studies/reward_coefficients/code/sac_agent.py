"""
Explicit PyTorch SAC (standard formulation) for the source-domain screening.

No external RL library. Networks mirror the DDPG source widths:
  actor  : state_dim -> 4096 -> 2048 -> 1024 -> 512 -> (mu, log_std)
  critic : (state, action) -> 4096 -> 2048 -> 1024 -> 512 -> 1  (twin Q1/Q2)

Squashed-Gaussian policy with tanh Jacobian log-prob correction, twin target
critics (no target actor), automatic entropy tuning.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim=1,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.mu_head = nn.Linear(512, action_dim)
        self.log_std_head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x):
        """Return (squashed action, log_prob with tanh correction, tanh(mu))."""
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()                       # reparameterization
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, s, a, ns, r, d):
        self.memory.append((s, a, ns, r, d))

    def sample(self, batch_size, device):
        batch = random.sample(self.memory, batch_size)
        s, a, ns, r, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
        a = torch.tensor(np.array(a), dtype=torch.float32, device=device).view(-1, 1)
        ns = torch.tensor(np.array(ns), dtype=torch.float32, device=device)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=device).view(-1, 1)
        d = torch.tensor(np.array(d), dtype=torch.float32, device=device).view(-1, 1)
        return s, a, ns, r, d

    def __len__(self):
        return len(self.memory)


class SACAgent:
    def __init__(self, state_dim, device, cfg):
        self.device = device
        self.gamma = cfg["train"]["gamma"]
        self.tau = cfg["train"]["tau"]
        self.reward_scale = cfg["sac"]["reward_scale"]
        self.autotune = cfg["sac"]["autotune_alpha"]

        self.actor = GaussianActor(state_dim, 1,
                                   cfg["sac"]["log_std_min"], cfg["sac"]["log_std_max"]).to(device)
        self.q1 = QNet(state_dim).to(device)
        self.q2 = QNet(state_dim).to(device)
        self.q1_target = QNet(state_dim).to(device)
        self.q2_target = QNet(state_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        wd = cfg["sac"]["weight_decay"]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["sac"]["actor_lr"], weight_decay=wd)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg["sac"]["critic_lr"], weight_decay=wd)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg["sac"]["critic_lr"], weight_decay=wd)

        self.target_entropy = float(cfg["sac"]["target_entropy"])
        init_alpha = float(cfg["sac"]["init_alpha"])
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                      device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["sac"]["alpha_lr"])

        self.replay = ReplayMemory(cfg["train"]["memory_capacity"])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_t(self, states):
        if not torch.is_tensor(states):
            states = torch.from_numpy(np.asarray(states)).float().to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        return states

    def select_action(self, states):
        """Stochastic action for exploration/rollout. Returns np array shape (1,)."""
        s = self._to_t(states)
        with torch.no_grad():
            a, _, _ = self.actor.sample(s)
        return a.cpu().numpy().reshape(-1)

    def deterministic_action(self, states):
        """tanh(mu) mean action. Returns np array shape (1,)."""
        s = self._to_t(states)
        with torch.no_grad():
            _, _, mean_a = self.actor.sample(s)
        return mean_a.cpu().numpy().reshape(-1)

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return None
        s, a, ns, r, d = self.replay.sample(batch_size, self.device)
        r = r * self.reward_scale

        # ---- critic target ----
        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(ns)
            q1t = self.q1_target(ns, na)
            q2t = self.q2_target(ns, na)
            min_qt = torch.min(q1t, q2t) - self.alpha * nlogp
            y = r + (1.0 - d) * self.gamma * min_qt

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # ---- actor ----
        a_new, logp, _ = self.actor.sample(s)
        q1_pi = self.q1(s, a_new)
        q2_pi = self.q2(s, a_new)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - min_q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ---- temperature ----
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # ---- soft update targets ----
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()), "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()), "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_logp": float(logp.mean().item()),
            "entropy": float(-logp.mean().item()),
        }

    def save(self, paths):
        torch.save(self.actor.state_dict(), paths["actor"])
        torch.save(self.q1.state_dict(), paths["q1"])
        torch.save(self.q2.state_dict(), paths["q2"])
        torch.save(self.q1_target.state_dict(), paths["q1_target"])
        torch.save(self.q2_target.state_dict(), paths["q2_target"])
        torch.save({"log_alpha": self.log_alpha.detach().cpu()}, paths["log_alpha"])
