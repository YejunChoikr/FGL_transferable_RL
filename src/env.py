"""Design environment and the convolutional surrogate it queries.

An episode assigns one thickness to each of the 30 independent cells, in a
row-wise raster over the half domain. The state holds the thicknesses assigned
so far together with their positional encodings, giving 60 entries. The
surrogate is queried once, at the terminal step, to predict the nine probe
displacements from the completed thickness field.

The state and action representation is the same in every domain. Actions lie on
[-1, 1] and the mapping onto millimetres differs by domain (1.5 + 0.3a for
[1.2, 1.8] mm, 1.8 + 0.6a for [1.2, 2.4] mm); that mapping belongs to the
surrogate trained on the domain, not to the observation.
"""
from __future__ import annotations

import math
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn

STATE_ENCODING = "surrogate_interleaved_thickness_location"


def location_grid(rows=10, cols=3, col_step=1):
    r = 5.0 * (np.arange(rows) + 1)[:, None]
    c = 5.0 * (col_step * np.arange(cols) + 1)[None, :]
    norm = math.sqrt((rows * 5.0) ** 2 + (col_step * (cols - 1) * 5.0 + 5.0) ** 2)
    return np.sqrt(r ** 2 + c ** 2) / norm


class OutputScaler:
    """Per-probe standardization of the surrogate output.

    The surrogate is trained on standardized displacements. ``mean`` and
    ``scale`` are read from the JSON file shipped beside each set of weights.
    """

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.scale = np.asarray(d["scale"], dtype=np.float64)

    def inverse_transform(self, y):
        return np.asarray(y, dtype=np.float64) * self.scale + self.mean


class SurrogateCNN(nn.Module):
    """Convolutional surrogate: thickness field to nine probe displacements."""

    def __init__(self, output_dim, leakyrelu_para):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(leakyrelu_para),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(leakyrelu_para),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(leakyrelu_para),
            nn.Linear(128, 64), nn.LeakyReLU(leakyrelu_para),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = x.view(-1, 1, 10, 6)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Environment:
    """Half-domain 10 x 3 thickness-assignment environment, terminal-only reward."""

    def __init__(self, case, model_row, model_col, surrogate_path, y_scaler_path,
                 reward_function, device, leakyrelu_para=0.005, domain="source"):
        self.device = device
        self.domain = domain
        self.surr = SurrogateCNN(model_row - 1, leakyrelu_para).to(device)
        self.surr.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=True))
        self.surr.eval()
        for p in self.surr.parameters():
            p.requires_grad_(False)

        self.case = case
        self.reward_function = reward_function
        self.y_scaler_path = y_scaler_path
        self.surrogate_path = surrogate_path

        self.x = 0
        self.y = 0
        self.epi_count = 0
        self.row = model_row
        self.col = model_col
        self.hw = self.row * self.col
        self.state_dim = self.hw * 2
        self.observation_space = np.zeros((self.state_dim,), dtype=np.float32)

        self.s = np.zeros((self.hw,))
        self.location = np.zeros((self.hw,))
        self.all_state = np.zeros((self.state_dim,), dtype=np.float64)
        self.last_surrogate_input = None

        self._loc = location_grid(self.row, self.col)
        self.load_scaler_Y = OutputScaler(self.y_scaler_path)

    # ---------- core RL API ----------
    def reset(self):
        self.x = 0
        self.y = 0
        self.epi_count = 0
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
            reward, _ = self.eval_surr()
            return self._obs(), reward, self.done()
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
        solve_model = torch.tensor(self.all_state.flatten(),
                                   dtype=torch.float32).to(self.device)
        self.last_surrogate_input = solve_model.detach().cpu().numpy().copy()
        with torch.no_grad():
            surr_result = self.surr(solve_model).cpu().numpy()
        surr_result = self.load_scaler_Y.inverse_transform(surr_result.reshape(1, -1))
        xr = surr_result.flatten()
        model_density = np.sum((self.s + 2) / 2) / 30.0
        reward = self.reward_function(*xr) / model_density
        return reward, surr_result

    def move(self):
        if self.case != "raster":
            raise ValueError("unknown traversal %r" % self.case)
        self._raster()

    def _raster(self):
        if self.y < self.col - 1:
            self.y += 1
        elif self.y == self.col - 1:
            self.x += 1
            self.y = 0
        self.epi_count += 1

    def state_update(self, action):
        self.s = self.s.reshape(self.row, self.col)
        self.s[self.x, self.y] = action.item()
        self.s = self.s.flatten()

    def location_information(self):
        self.location = self.location.reshape(self.row, self.col)
        self.location[self.x, self.y] = self._loc[self.x, self.y]

    def assembly_location(self):
        self.s = self.s.reshape(self.row, self.col)
        self.all_state = np.stack(
            (self.s, np.broadcast_to(self.location, self.s.shape)), axis=2
        ).reshape(-1)
        self.s = self.s.flatten()
        return self.all_state

    def _obs(self):
        return self.all_state.astype(np.float32)

    # ---------- helpers ----------
    def surrogate_profile(self):
        solve_model = torch.tensor(self.all_state.flatten(),
                                   dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.surr(solve_model).cpu().numpy()
        return self.load_scaler_Y.inverse_transform(out.reshape(1, -1)).flatten()

    def density(self):
        return float(np.sum((self.s + 2) / 2) / 30.0)

    def actions_row_major(self):
        """The 30 placed actions in the 10x3 row-major physical order."""
        return self.s.reshape(-1).astype(np.float64).copy()

    def evaluate_action_vector(self, actions):
        """Evaluate a complete 30-action design directly (no stepping)."""
        a = np.asarray(actions, dtype=np.float64).reshape(self.row, self.col)
        loc = np.zeros((self.row, self.col))
        loc_max = math.sqrt((self.row * 5) ** 2 + (self.col * 5) ** 2)
        for i in range(self.row):
            for j in range(self.col):
                loc[i, j] = math.sqrt(((i + 1) * 5) ** 2 + ((j + 1) * 5) ** 2) / loc_max
        self.s = a.flatten()
        # assembly_location() broadcasts self.location against the (row, col) state,
        # so location must stay 2-D here exactly as location_information() leaves it
        self.location = loc
        self.assembly_location()
        return self.eval_surr()


def configure_determinism():
    """Disable reduced-precision paths so three different GPUs agree numerically."""
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


def make_env(task, domain, cfg, root, device):
    """Build the Environment for one FGL task in one domain."""
    from pathlib import Path
    from reward import REWARDS, TASK_TO_REWARD
    root = Path(root)
    d = cfg["domains"][domain]
    return Environment(
        case=cfg["env"]["case"],
        model_row=cfg["env"]["model_row"],
        model_col=cfg["env"]["model_col"],
        surrogate_path=str(root / cfg["assets"][d["surrogate"]]),
        y_scaler_path=str(root / cfg["assets"][d["scaler"]]),
        reward_function=REWARDS[TASK_TO_REWARD[task]],
        device=device,
        leakyrelu_para=cfg["env"]["leakyrelu_para"],
        domain=domain,
    )
