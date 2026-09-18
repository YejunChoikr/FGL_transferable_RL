"""Source-domain Environment with a coefficient-parameterised reward.

Identical mechanics and 60-dim state to the calibration/go-no-go environment; the
only change is that ``eval_surr`` divides the numerator by ``W(C2, rho_A)`` instead
of by the fixed density code, so that (C1, C2) can be swept.

Source domain only: surrogate_6x10_30000_1.2_to_1.8. No transfer, no warm start.
"""
from __future__ import annotations

import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from reward_cc import reward as reward_fn, rho_a, weight_W

STATE_ENCODING = "surrogate_interleaved_thickness_location"


class SurrogateCNN(nn.Module):
    """Unchanged from the verified source implementation."""

    def __init__(self, output_dim, leakyrelu_para):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(leakyrelu_para),
            nn.Linear(128, 64), nn.LeakyReLU(leakyrelu_para),
            nn.Linear(64, output_dim))

    def forward(self, x):
        x = x.view(1, 1, 10, 6)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.view(x.size(0), -1))


class Environment:
    def __init__(self, case, model_row, model_col, surrogate_path, y_scaler_path,
                 task, c1, c2, device, leakyrelu_para=0.005):
        self.device = device
        self.task, self.c1, self.c2 = task, float(c1), float(c2)
        self.surr = SurrogateCNN(model_row - 1, leakyrelu_para).to(device)
        self.surr.load_state_dict(torch.load(surrogate_path, map_location=device))
        self.surr.eval()
        for p in self.surr.parameters():
            p.requires_grad_(False)

        self.case = case
        self.row, self.col = model_row, model_col
        self.hw = self.row * self.col
        self.state_dim = self.hw * 2
        self.observation_space = np.zeros((self.state_dim,), dtype=np.float32)
        self.x = self.y = self.epi_count = 0
        self.s = np.zeros((self.hw,))
        self.location = np.zeros((self.hw,))
        self.all_state = np.zeros((self.state_dim,), dtype=np.float64)
        self.last_surrogate_input = None
        with open(y_scaler_path, "rb") as f:
            self.load_scaler_Y = pickle.load(f)

    # ---------- core RL API ----------
    def reset(self):
        self.x = self.y = self.epi_count = 0
        self.s = np.zeros((self.hw,))
        self.location = np.zeros((self.hw,))
        self.all_state = np.zeros((self.state_dim,), dtype=np.float64)
        self.last_surrogate_input = None
        return self._obs()

    def step(self, action):
        return self.replay_step(action)

    def replay_step(self, action):
        if self.done():
            self.state_update(action)
            self.location_information()
            self.assembly_location()
            r, _ = self.eval_surr()
            return self._obs(), r, self.done()
        self.state_update(action)
        self.location_information()
        self.assembly_location()
        done = self.done()
        self.move()
        return self._obs(), 0, done

    def done(self):
        return self.epi_count == self.hw - 1

    # ---------- internals ----------
    def eval_surr(self):
        t = torch.tensor(self.all_state.flatten(), dtype=torch.float32).to(self.device)
        self.last_surrogate_input = t.detach().cpu().numpy().copy()
        with torch.no_grad():
            out = self.surr(t).cpu().numpy()
        surr_result = self.load_scaler_Y.inverse_transform(out.reshape(1, -1))
        xr = surr_result.flatten()
        # ONE implementation: the training signal and every analysis metric come from
        # reward_cc.reward, so the two can never diverge. (Computing the numerator in a
        # local float32 closure instead differed from the float64 analysis path by ~7e-8,
        # because the scaler returns float32 and NumPy value-based casting preserved it.)
        return reward_fn(xr, self.s, self.task, self.c1, self.c2), surr_result

    def move(self):
        if self.case == "zigzag":
            if self.y < self.col - 1:
                self.y += 1
            else:
                self.x += 1
                self.y = 0
            self.epi_count += 1

    def state_update(self, action):
        self.s = self.s.reshape(self.row, self.col)
        self.s[self.x, self.y] = action.item()
        self.s = self.s.flatten()

    def location_information(self):
        self.location = self.location.reshape(self.row, self.col)
        loc_max = math.sqrt((self.row * 5) ** 2 + (self.col * 5) ** 2)
        self.location[self.x, self.y] = math.sqrt(((self.x + 1) * 5) ** 2 +
                                                  ((self.y + 1) * 5) ** 2) / loc_max

    def assembly_location(self):
        self.s = self.s.reshape(self.row, self.col)
        self.all_state = np.stack(
            (self.s, np.broadcast_to(self.location, self.s.shape)), axis=2).reshape(-1)
        self.s = self.s.flatten()
        return self.all_state

    def _obs(self):
        return self.all_state.astype(np.float32)

    # ---------- helpers ----------
    def surrogate_profile(self):
        t = torch.tensor(self.all_state.flatten(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.surr(t).cpu().numpy()
        return self.load_scaler_Y.inverse_transform(out.reshape(1, -1)).flatten()

    def actions_row_major(self):
        return self.s.reshape(-1).astype(np.float64).copy()

    def rho_A(self):
        return rho_a(self.s)

    def W(self):
        return weight_W(self.c2, rho_a(self.s))

    def evaluate_action_vector(self, actions):
        a = np.asarray(actions, dtype=np.float64).reshape(self.row, self.col)
        loc = np.zeros((self.row, self.col))
        loc_max = math.sqrt((self.row * 5) ** 2 + (self.col * 5) ** 2)
        for i in range(self.row):
            for j in range(self.col):
                loc[i, j] = math.sqrt(((i + 1) * 5) ** 2 + ((j + 1) * 5) ** 2) / loc_max
        self.s = a.flatten()
        self.location = loc          # assembly_location broadcasts against (row, col)
        self.assembly_location()
        return self.eval_surr()


def configure_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    configure_determinism()


def make_env(task, c1, c2, cfg, root, device):
    from pathlib import Path
    root = Path(root)
    return Environment(
        case=cfg["env"]["case"], model_row=cfg["env"]["model_row"],
        model_col=cfg["env"]["model_col"],
        surrogate_path=str(root / cfg["assets"]["surrogate"]),
        y_scaler_path=str(root / cfg["assets"]["scaler_Y"]),
        task=task, c1=c1, c2=c2, device=device,
        leakyrelu_para=cfg["env"]["leakyrelu_para"])
