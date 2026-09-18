"""GPU-resident uniform replay buffer with terminal relabelling fields.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .protocol import load_protocol

_POLICY = load_protocol()["policy"]

CAPACITY: int = int(_POLICY["replay_capacity"])
SAMPLING_LAW: str = str(_POLICY["replay_sampling"])
STATE_DIM: int = int(load_protocol()["encoding"]["complete_state_dim"])
N_PROBES: int = 9
N_CELLS: int = 30

FIELDS = ("s", "a", "r", "s2", "done", "u9", "actions30", "goal", "omega")
_WIDTHS = {"s": STATE_DIM, "a": 1, "r": 1, "s2": STATE_DIM, "done": 1,
           "u9": N_PROBES, "actions30": N_CELLS, "goal": 1, "omega": 1}


class GPUReplay:
    """Pre-allocated float32 ring buffer that lives on the training device."""

    def __init__(self, capacity: int = CAPACITY, device="cuda",
                 state_dim: int = STATE_DIM):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self._pos = 0
        self._size = 0
        self._buf = {
            k: torch.zeros((self.capacity, _WIDTHS[k]), dtype=torch.float32,
                           device=self.device)
            for k in FIELDS
        }

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._pos

    def field(self, name: str) -> torch.Tensor:
        """Return the whole backing tensor of one field (no copy)."""
        return self._buf[name]

    # ------------------------------------------------------------------ write
    def add(self, s, a, r, s2, done, u9=None, actions30=None, goal=None) -> None:
        """Append a batch of ``B`` transitions.

        Every argument is a device tensor whose leading dimension is ``B``.
        ``u9`` and ``actions30`` may be ``None`` for non-terminal steps, in
        which case zeros are stored (their relabelled reward is zero anyway
        because ``done`` is zero).
        """
        b = int(s.shape[0])
        if b > self.capacity:
            raise ValueError("batch larger than replay capacity")
        vals = {
            "s": s.reshape(b, self.state_dim),
            "a": a.reshape(b, 1),
            "r": r.reshape(b, 1),
            "s2": s2.reshape(b, self.state_dim),
            "done": done.reshape(b, 1),
            "u9": (torch.zeros((b, N_PROBES), dtype=torch.float32,
                               device=self.device) if u9 is None
                   else u9.reshape(b, N_PROBES)),
            "actions30": (torch.zeros((b, N_CELLS), dtype=torch.float32,
                                      device=self.device) if actions30 is None
                          else actions30.reshape(b, N_CELLS)),
            "goal": (torch.zeros((b, 1), dtype=torch.float32,
                                 device=self.device) if goal is None
                     else goal.reshape(b, 1)),
        }
        vals["omega"] = (vals["goal"] - 5.0) / 2.0
        end = self._pos + b
        if end <= self.capacity:
            # Contiguous fast path; the production run stores 45,000 of the
            # 100,000 slots and therefore never wraps.
            for k in FIELDS:
                self._buf[k][self._pos:end].copy_(vals[k].to(torch.float32))
        else:
            idx = ((torch.arange(b, device=self.device) + self._pos)
                   % self.capacity)
            for k in FIELDS:
                self._buf[k].index_copy_(0, idx, vals[k].to(torch.float32))
        self._pos = int(end % self.capacity)
        self._size = int(min(self._size + b, self.capacity))

    # ----------------------------------------------------------------- sample
    def sample_indices(self, batch_size: int,
                       generator: Optional[torch.Generator] = None
                       ) -> torch.Tensor:
        """Uniform WITHOUT replacement over the currently valid entries.

        ``torch.randperm`` over the valid range and taking the first
        ``batch_size`` entries is exactly a uniform draw without replacement,
        which is the sampling law fixed by protocol.policy.replay_sampling.
        """
        n = self._size
        if batch_size > n:
            raise ValueError("requested %d samples from %d valid entries"
                             % (batch_size, n))
        perm = torch.randperm(n, generator=generator, device=self.device)
        return perm[:batch_size]

    def gather(self, idx: torch.Tensor) -> dict:
        """Return the stored fields at ``idx`` as a dict of device tensors."""
        idx = idx.to(device=self.device, dtype=torch.long)
        out = {k: self._buf[k].index_select(0, idx) for k in FIELDS}
        out["idx"] = idx
        return out

    def sample(self, batch_size: int,
               generator: Optional[torch.Generator] = None) -> dict:
        """Draw one minibatch (uniform, without replacement)."""
        return self.gather(self.sample_indices(batch_size, generator))

    # -------------------------------------------------------------- reference
    def to_numpy(self, n: Optional[int] = None) -> dict:
        """Copy the valid entries to CPU numpy arrays (reference comparisons)."""
        m = self._size if n is None else int(n)
        return {k: self._buf[k][:m].detach().cpu().numpy().copy()
                for k in FIELDS}

    def reference_gather(self, idx) -> dict:
        """CPU numpy gather with the same indices, for the gate comparison."""
        arr = self.to_numpy()
        i = np.asarray(idx.detach().cpu().numpy() if torch.is_tensor(idx)
                       else idx, dtype=np.int64)
        return {k: arr[k][i] for k in FIELDS}

    def compare_with_reference(self, idx) -> dict:
        """Max absolute deviation between the GPU gather and the CPU gather."""
        gpu = self.gather(idx if torch.is_tensor(idx)
                          else torch.as_tensor(idx, dtype=torch.long))
        ref = self.reference_gather(idx)
        diffs = {}
        for k in FIELDS:
            g = gpu[k].detach().cpu().numpy()
            diffs[k] = float(np.max(np.abs(g - ref[k]))) if g.size else 0.0
        return {"max_abs_diff": diffs,
                "identical": bool(all(v == 0.0 for v in diffs.values()))}

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        """Full resume state (device tensors moved to CPU)."""
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "pos": self._pos,
            "size": self._size,
            "buf": {k: self._buf[k][:self._size].detach().cpu().clone()
                    for k in FIELDS},
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore a resume state written by :meth:`state_dict`."""
        if int(sd["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch on resume")
        self._pos = int(sd["pos"])
        self._size = int(sd["size"])
        for k in FIELDS:
            v = sd["buf"][k]
            self._buf[k].zero_()
            self._buf[k][:v.shape[0]].copy_(v.to(self.device))


def sampling_histogram(replay: GPUReplay, batch_size: int, draws: int,
                       generator: Optional[torch.Generator] = None
                       ) -> np.ndarray:
    """Count how often each valid entry is drawn over ``draws`` minibatches."""
    n = len(replay)
    counts = np.zeros(n, dtype=np.int64)
    for _ in range(int(draws)):
        idx = replay.sample_indices(batch_size, generator)
        np.add.at(counts, idx.detach().cpu().numpy(), 1)
    return counts


def chi_square_uniform(counts: np.ndarray, batch_size: int = None) -> dict:
    """Pearson chi-square goodness-of-fit against the uniform draw law.

    Sampling is uniform WITHOUT replacement inside each minibatch, so an entry's
    count over ``D`` minibatches is a sum of Bernoulli(n/N) indicators and its
    variance is ``E * (1 - n/N)``, not the multinomial ``E``. Passing
    ``batch_size = n`` applies that finite-population correction; without it the
    statistic is deflated and a correct sampler looks "too uniform".
    """
    c = np.asarray(counts, dtype=np.float64)
    n_bins = c.size
    total = float(c.sum())
    expected = total / n_bins
    raw = float(((c - expected) ** 2 / expected).sum())
    if batch_size is None:
        factor = 1.0
    else:
        factor = 1.0 - float(batch_size) / float(n_bins)
        if factor <= 0:
            raise ValueError("batch_size must be smaller than the valid range")
    stat = raw / factor
    dof = n_bins - 1
    # Wilson-Hilferty normal approximation; no SciPy dependency in the runtime.
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(
        2.0 / (9.0 * dof))
    return {"chi2_raw": raw, "finite_population_factor": factor,
            "chi2": stat, "dof": int(dof), "expected_per_bin": expected,
            "z_wilson_hilferty": float(z), "n_draws": total}


def within_batch_unique(idx: torch.Tensor) -> bool:
    """True when a drawn minibatch contains no repeated index."""
    return int(torch.unique(idx).numel()) == int(idx.numel())


__all__ = ["GPUReplay", "CAPACITY", "SAMPLING_LAW", "FIELDS",
           "sampling_histogram", "chi_square_uniform", "within_batch_unique"]
