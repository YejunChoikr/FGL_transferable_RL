"""Initializing a target-domain agent from a source-domain agent.

The four hidden layers of the actor and of each critic are copied; output layers
keep their target-domain initialization. Target networks are set to copies of
their own online networks once assembly is complete.
"""
from __future__ import annotations

HIDDEN_LAYERS = ("fc1", "fc2", "fc3", "fc4")


def _copy_hidden(module, source_state, label, moved):
    """Copy the hidden-layer tensors of one network, checking every shape."""
    dst = module.state_dict()
    for key in list(dst):
        if key.split(".")[0] not in HIDDEN_LAYERS or key not in source_state:
            continue
        if dst[key].shape != source_state[key].shape:
            raise ValueError("shape mismatch on %s.%s: %s vs %s"
                             % (label, key, tuple(dst[key].shape),
                                tuple(source_state[key].shape)))
        dst[key] = source_state[key].clone()
        moved.append("%s:%s" % (label, key))
    module.load_state_dict(dst)


def _check(moved, n_networks):
    expected = 2 * len(HIDDEN_LAYERS) * n_networks
    if len(moved) != expected:
        raise RuntimeError("transferred %d tensors, expected %d: %s"
                           % (len(moved), expected, moved))


def transfer_ddpg(agent, source_state):
    """Actor and critic hidden layers: 16 tensors.

    ``source_state`` holds the ``actor`` and ``critic`` state dictionaries of
    the source agent.
    """
    moved = []
    _copy_hidden(agent.actor, source_state["actor"], "actor", moved)
    _copy_hidden(agent.critic, source_state["critic"], "critic", moved)
    _check(moved, 2)
    agent.hard_update()
    return moved


def transfer_sac(agent, source_state):
    """Actor and both online critic hidden layers: 24 tensors.

    ``source_state`` holds the ``actor``, ``q1`` and ``q2`` state dictionaries
    of the source agent.
    """
    moved = []
    _copy_hidden(agent.actor, source_state["actor"], "actor", moved)
    _copy_hidden(agent.q1, source_state["q1"], "q1", moved)
    _copy_hidden(agent.q2, source_state["q2"], "q2", moved)
    _check(moved, 3)
    agent.q1_target.load_state_dict(agent.q1.state_dict())
    agent.q2_target.load_state_dict(agent.q2.state_dict())
    return moved
