"""Initial policy tensors, the goal-column widening and the transfer loader.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from . import models_policy as M
from .paths import INIT_BANK, LOCKS
from .protocol import load_protocol
from .rng import numpy_gen, subseed

_POLICY = load_protocol()["policy"]
_RT = load_protocol()["runtime"]

PREFILL_EPISODES: int = int(_POLICY["prefill_episodes_in_total"])
STEPS_PER_EPISODE: int = int(_POLICY["steps_per_episode"])
SEEDS: tuple = (0, 1, 2, 3, 4)

#: Network name -> rng stream name (protocol.runtime.rng_stream_ids).
INIT_STREAMS: dict = {
    ("SAC", "actor"): "sac_actor_init",
    ("SAC", "q1"): "sac_q1_init",
    ("SAC", "q2"): "sac_q2_init",
    ("DDPG", "actor"): "ddpg_actor_init",
    ("DDPG", "critic"): "ddpg_critic_init",
}
PREFILL_STREAM = "prefill"

#: init_bank file stem for each (algo, network).
FILE_STEMS: dict = {
    ("SAC", "actor"): "sac_actor",
    ("SAC", "q1"): "sac_q1",
    ("SAC", "q2"): "sac_q2",
    ("DDPG", "actor"): "ddpg_actor",
    ("DDPG", "critic"): "ddpg_critic",
}


# --------------------------------------------------------------------- hashing
def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of a CPU float32 tensor's raw contiguous bytes."""
    a = t.detach().cpu().contiguous().to(torch.float32).numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def state_dict_sha256(sd: dict) -> str:
    """Order-independent digest of a state_dict (key + shape + tensor bytes)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(str(tuple(sd[k].shape)).encode())
        h.update(sd[k].detach().cpu().contiguous().to(torch.float32)
                 .numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of the raw bytes of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_state(net: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ---------------------------------------------------------------- construction
def build_policy_init(seed: int, algo: str, shared: bool = False) -> dict:
    """Build the pinned CPU initial tensors for one seed and algorithm.

    Returns ``{'actor': state_dict, 'q1': ..., 'q2': ...}`` for SAC and
    ``{'actor': ..., 'critic': ...}`` for DDPG. Each network is constructed
    inside a forked CPU RNG seeded with its own protocol sub-seed, so the global
    RNG of the calling process is untouched and the tensors depend only on
    ``(seed, stream_id)``.

    ``shared=True`` returns the goal-conditioned tensors, i.e. the same scratch
    state with ``fc1.weight`` widened by one zero column.
    """
    algo = str(algo).upper()
    if algo == "SAC":
        nets = ("actor", "q1", "q2")
    elif algo == "DDPG":
        nets = ("actor", "critic")
    else:
        raise ValueError("algo must be SAC or DDPG, got %r" % algo)

    out = {}
    for name in nets:
        stream = INIT_STREAMS[(algo, name)]
        s = subseed(int(seed), stream)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(s))
            if name == "actor":
                net = M.make_actor(algo, shared=False)
            else:
                net = M.make_critic(shared=False)
        sd = _cpu_state(net)
        out[name] = widen_fc1_with_zero_goal_column(sd) if shared else sd
    return out


def build_prefill(seed: int) -> np.ndarray:
    """Return the stored ``(20, 30)`` uniform[-1,1] prefill action array.

    One array per seed, shared across domains, budgets, algorithms and arms
    (protocol.runtime.prefill_pairing). Stored as float32 because the
    environment consumes float32 actions.
    """
    g = numpy_gen(subseed(int(seed), PREFILL_STREAM))
    a = g.uniform(-1.0, 1.0, size=(PREFILL_EPISODES, STEPS_PER_EPISODE))
    return np.asarray(a, dtype=np.float32)


def widen_fc1_with_zero_goal_column(state_dict: dict) -> dict:
    """Append one zero input column to ``fc1.weight``; copy everything else.

    The goal coordinate is the last input column of the actor and of each
    critic, so a zero column makes the widened network compute exactly the
    target-specific function at every goal before training starts.
    """
    if "fc1.weight" not in state_dict:
        raise KeyError("state_dict has no fc1.weight")
    out = {}
    for k, v in state_dict.items():
        t = v.detach().cpu().clone()
        if k == "fc1.weight":
            widened = torch.zeros((t.shape[0], t.shape[1] + M.GOAL_DIM),
                                  dtype=t.dtype)
            widened[:, :t.shape[1]] = t
            out[k] = widened
        else:
            out[k] = t
    return out


def widen_report(original: dict, widened: dict) -> dict:
    """Audit record proving the widening only added a zero column."""
    n_old = int(original["fc1.weight"].shape[1])
    n_new = int(widened["fc1.weight"].shape[1])
    same = {k: bool(torch.equal(original[k], widened[k]))
            for k in original if k != "fc1.weight"}
    return {
        "fc1_in_specific": n_old,
        "fc1_in_shared": n_new,
        "added_columns": n_new - n_old,
        "new_column_is_zero": bool(
            torch.count_nonzero(widened["fc1.weight"][:, n_old:]) == 0),
        "fc1_leading_block_identical": bool(
            torch.equal(widened["fc1.weight"][:, :n_old],
                        original["fc1.weight"])),
        "other_keys_identical": same,
        "all_other_keys_identical": bool(all(same.values())),
    }


# ------------------------------------------------------------------- init bank
def init_path(algo: str, net: str, seed: int, root: Path = None) -> Path:
    """Path of one stored initial state_dict inside ``init_bank/``."""
    base = INIT_BANK if root is None else Path(root)
    return base / ("%s_s%d.pt" % (FILE_STEMS[(str(algo).upper(), net)],
                                  int(seed)))


def prefill_path(seed: int, root: Path = None) -> Path:
    """Path of one stored prefill action array inside ``init_bank/``."""
    base = INIT_BANK if root is None else Path(root)
    return base / ("prefill_s%d.npy" % int(seed))


def load_policy_init(algo: str, net: str, seed: int, root: Path = None) -> dict:
    """Load one stored initial state_dict (CPU tensors)."""
    p = init_path(algo, net, seed, root)
    return torch.load(p, map_location="cpu", weights_only=True)


def load_prefill(seed: int, root: Path = None) -> np.ndarray:
    """Load the stored ``(20, 30)`` prefill action array for one seed."""
    a = np.load(prefill_path(seed, root))
    if a.shape != (PREFILL_EPISODES, STEPS_PER_EPISODE):
        raise ValueError("prefill array has shape %s" % (a.shape,))
    return np.asarray(a, dtype=np.float32)


def _save_atomic(obj, path: Path, saver) -> None:
    """Write through a temporary sibling and rename, so readers never see a
    partial file. The temporary keeps the original suffix because ``np.save``
    appends ``.npy`` to any other name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    saver(obj, tmp)
    tmp.replace(path)


def write_init_bank(seeds: Sequence[int] = SEEDS,
                    root: Path = None) -> dict:
    """Generate and store every policy initial tensor and prefill array.

    Returns the manifest that becomes ``INIT_LOCK['policy']``.
    """
    base = INIT_BANK if root is None else Path(root)
    base.mkdir(parents=True, exist_ok=True)
    files: dict = {}

    for seed in seeds:
        for algo in ("SAC", "DDPG"):
            blob = build_policy_init(seed, algo, shared=False)
            for net, sd in blob.items():
                p = init_path(algo, net, seed, base)
                _save_atomic(sd, p, lambda o, q: torch.save(o, q))
                shared_sd = widen_fc1_with_zero_goal_column(sd)
                files[p.name] = {
                    "algo": algo,
                    "net": net,
                    "seed": int(seed),
                    "stream": INIT_STREAMS[(algo, net)],
                    "subseed": int(subseed(int(seed),
                                           INIT_STREAMS[(algo, net)])),
                    "file_sha256": file_sha256(p),
                    "state_dict_sha256": state_dict_sha256(sd),
                    "shared_state_dict_sha256": state_dict_sha256(shared_sd),
                    "keys": sorted(sd),
                    "shapes": {k: list(v.shape) for k, v in sd.items()},
                    "tensor_sha256": {k: tensor_sha256(v)
                                      for k, v in sd.items()},
                    "n_params_specific": int(sum(v.numel()
                                                 for v in sd.values())),
                    "n_params_shared": int(sum(v.numel()
                                               for v in shared_sd.values())),
                }
        pre = build_prefill(seed)
        pp = prefill_path(seed, base)
        _save_atomic(pre, pp,
                     lambda o, q: np.save(str(q), o, allow_pickle=False))
        files[pp.name] = {
            "kind": "prefill",
            "seed": int(seed),
            "stream": PREFILL_STREAM,
            "subseed": int(subseed(int(seed), PREFILL_STREAM)),
            "file_sha256": file_sha256(pp),
            "array_sha256": hashlib.sha256(pre.tobytes()).hexdigest(),
            "shape": list(pre.shape),
            "dtype": str(pre.dtype),
            "min": float(pre.min()),
            "max": float(pre.max()),
        }

    return {
        "generated_by": "cmame_rt.policy_init.write_init_bank",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seeds": [int(s) for s in seeds],
        "rule": {
            "subseed": _RT["rng_subseed_rule"],
            "initialization": _RT["initialization"],
            "initial_tensor_pairing": _RT["initial_tensor_pairing"],
            "prefill_pairing": _RT["prefill_pairing"],
            "construction": ("torch.random.fork_rng + torch.manual_seed"
                             "(subseed) on CPU, then default "
                             "nn.Linear.reset_parameters"),
            "shared_from_specific": ("widen_fc1_with_zero_goal_column; no "
                                     "separate goal-conditioned draw"),
            "prefill_dtype": "float32",
            "reproducible_on_any_host": True,
            "regeneration": ("cmame_rt.policy_init.write_init_bank reproduces "
                             "every tensor bit-for-bit from (seed, stream_id); "
                             "a remote host may regenerate instead of copying "
                             "1.1 GB, then verify against this manifest"),
        },
        "streams": {"%s.%s" % (a, n): INIT_STREAMS[(a, n)]
                    for (a, n) in INIT_STREAMS},
        "param_counts": param_count_table(),
        "input_dims": input_dim_table(),
        "files": files,
    }


def param_count_table() -> dict:
    """Trainable parameter count of every policy network variant."""
    out = {}
    for algo in ("SAC", "DDPG"):
        for shared in (False, True):
            tag = "shared" if shared else "specific"
            out["%s_actor_%s" % (algo.lower(), tag)] = M.count_params(
                M.make_actor(algo, shared))
            out["%s_critic_%s" % (algo.lower(), tag)] = M.count_params(
                M.make_critic(shared))
    return out


def input_dim_table() -> dict:
    """Actor and critic input widths for both conditioning modes."""
    return {
        "actor_specific": M.actor_in_dim(False),
        "actor_shared": M.actor_in_dim(True),
        "critic_specific": M.critic_in_dim(False),
        "critic_shared": M.critic_in_dim(True),
        "specific_actor_order": load_protocol()["encoding"]["specific_actor_order"],
        "specific_critic_order": load_protocol()["encoding"]["specific_critic_order"],
        "shared_actor_order": load_protocol()["encoding"]["shared_actor_order"],
        "shared_critic_order": load_protocol()["encoding"]["shared_critic_order"],
    }


# -------------------------------------------------------------------- transfer
def _transfer_keys() -> tuple:
    """``fc1..fc4`` weight/bias, taken from protocol.policy.transfer."""
    cfg = _POLICY["transfer"]
    actor = tuple(cfg["actor_copy"])
    critic = tuple(cfg["critic_copy"])
    if actor != ("fc1", "fc2", "fc3", "fc4") or critic != actor:
        raise RuntimeError("protocol.policy.transfer changed: %s / %s"
                           % (actor, critic))
    return M.TRUNK_KEYS


TRANSFER_KEYS: tuple = _transfer_keys()


def copy_trunk_(dst: torch.nn.Module, src_state: dict, label: str) -> dict:
    """Copy ``fc1..fc4`` weight/bias from ``src_state`` into ``dst`` in place.

    Shapes must match exactly; a missing or mis-shaped key stops the run rather
    than silently leaving a scratch tensor in place.
    """
    cur = dst.state_dict()
    moved = {}
    with torch.no_grad():
        for k in TRANSFER_KEYS:
            if k not in src_state:
                raise KeyError("[transfer] %s: source has no %s" % (label, k))
            if k not in cur:
                raise KeyError("[transfer] %s: target has no %s" % (label, k))
            s = src_state[k]
            if tuple(s.shape) != tuple(cur[k].shape):
                raise ValueError("[transfer] %s %s shape %s != %s"
                                 % (label, k, tuple(s.shape),
                                    tuple(cur[k].shape)))
            cur[k].copy_(s.to(device=cur[k].device, dtype=cur[k].dtype))
            moved[k] = {"shape": list(cur[k].shape),
                        "sha256": tensor_sha256(cur[k])}
    dst.load_state_dict(cur)
    return moved


def head_hashes(net: torch.nn.Module, kind: str, algo: str = "SAC") -> dict:
    """sha256 of the head tensors, which must stay at paired-scratch values."""
    keys = [k for k in M.expected_keys(kind, algo) if k not in M.TRUNK_KEYS]
    sd = net.state_dict()
    return {k: tensor_sha256(sd[k]) for k in keys}


def trunk_hashes(net: torch.nn.Module) -> dict:
    """sha256 of the transferable trunk tensors."""
    sd = net.state_dict()
    return {k: tensor_sha256(sd[k]) for k in TRANSFER_KEYS}


def _read_shared_json(path):
    """Read a lock file that another agent may have written.

    RUNTIME_LOCK and INIT_LOCK are shared, and a co-owner may write non-ASCII
    text. The bytes are decoded explicitly rather than through the platform
    default, which on this Windows host is cp949 and would raise. This agent
    always writes back with ensure_ascii, so it never introduces the problem.
    """
    raw = pathlib.Path(path).read_bytes().strip()
    if not raw:
        return {}
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("cannot decode %s as JSON" % path)


def write_lock(manifest: dict, locks_dir: Path = None) -> Path:
    """Merge ``manifest`` into ``locks/INIT_LOCK.json`` under the 'policy' key.

    INIT_LOCK is shared with the SURR agent, so the file is read, the 'policy'
    key replaced and everything else preserved.
    """
    base = LOCKS if locks_dir is None else Path(locks_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "INIT_LOCK.json"
    doc = _read_shared_json(path) if path.exists() else {}
    doc["policy"] = manifest
    tmp = path.with_suffix(".json.policytmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False, ensure_ascii=True)
        fh.write("\n")
    tmp.replace(path)
    return path


def verify_init_bank(manifest: dict, root: Path = None) -> dict:
    """Recheck the stored tensors against a manifest and against their seeds.

    The authoritative digest is the TENSOR CONTENT (``state_dict_sha256`` for a
    checkpoint, ``array_sha256`` for the prefill array), because that is the
    scientific object that has to be identical everywhere. The ``.pt``
    container bytes are reported separately: ``torch.save`` writes a zip whose
    non-tensor bytes are not guaranteed to be identical on another machine, so
    a container difference alongside identical tensors is recorded as an
    observation rather than a failure. A container difference WITH a tensor
    difference still fails, through the tensor check.
    """
    base = INIT_BANK if root is None else Path(root)
    problems = []
    container_differs = []
    checked = 0
    for name, rec in manifest["files"].items():
        p = base / name
        if not p.exists():
            problems.append("missing %s" % name)
            continue
        if file_sha256(p) != rec["file_sha256"]:
            container_differs.append(name)
        if rec.get("kind") == "prefill":
            a = np.load(p)
            rebuilt = build_prefill(rec["seed"])
            if hashlib.sha256(a.tobytes()).hexdigest() != rec["array_sha256"]:
                problems.append("prefill content mismatch %s" % name)
            if not np.array_equal(a, rebuilt):
                problems.append("prefill not reproducible %s" % name)
        else:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            if state_dict_sha256(sd) != rec["state_dict_sha256"]:
                problems.append("state_dict sha mismatch %s" % name)
            rebuilt = build_policy_init(rec["seed"], rec["algo"])[rec["net"]]
            if state_dict_sha256(rebuilt) != rec["state_dict_sha256"]:
                problems.append("init not reproducible %s" % name)
        checked += 1
    return {"n_checked": checked, "problems": problems,
            "ok": not problems,
            "tensor_content_matches_manifest": not problems,
            "container_bytes_differ": container_differs,
            "n_container_bytes_differ": len(container_differs),
            "container_note": ("torch.save zip bytes are host-local; the "
                               "tensor digests above are the authority and "
                               "they are recomputed from (seed, stream_id) as "
                               "well as read back from disk")}


__all__ = ["SEEDS", "PREFILL_EPISODES", "STEPS_PER_EPISODE", "INIT_STREAMS",
           "TRANSFER_KEYS", "build_policy_init", "build_prefill",
           "widen_fc1_with_zero_goal_column", "widen_report", "init_path",
           "prefill_path", "load_policy_init", "load_prefill",
           "write_init_bank", "write_lock", "verify_init_bank",
           "param_count_table", "input_dim_table", "copy_trunk_",
           "head_hashes", "trunk_hashes", "tensor_sha256",
           "state_dict_sha256", "file_sha256"]
