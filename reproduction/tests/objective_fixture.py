"""Random CNN fixture for optimizer integration and independent math checks."""
import numpy as np
import torch

from cmame_rt import encoding
from cmame_rt.direct import SurrogateObjective
from cmame_rt.models_surrogate import CNN
from cmame_rt.protocol import load_reference_math
from cmame_rt.reward import reward_torch


def build_pilot_objective(goal, device, seed=0):
    torch.manual_seed(seed)
    model = CNN().to(device).eval()
    model.requires_grad_(False)
    objective = SurrogateObjective(
        model=model, y_mean=torch.linspace(1., 5., 9, device=device),
        y_scale=torch.full((9,), .5, device=device), goal=goal,
        C1=.5, C2=1., device=device,
        encode_batch_torch=encoding.encode_batch_torch,
        to_cnn_input=encoding.to_cnn_input, reward_torch=reward_torch)
    return objective, sum(p.numel() for p in model.parameters())


def check_against_reference(objective):
    reference = load_reference_math()
    actions = np.random.default_rng(0).uniform(-1, 1, (4, 30)).astype(np.float32)
    rewards, u9 = objective.rewards(actions)
    expected = [reference.design_metrics(u, a.astype(np.float64), objective.goal,
                objective.C1, objective.C2)["reward"] for a, u in zip(actions, u9)]
    encoded = encoding.encode_batch_torch(torch.from_numpy(actions)).numpy()
    golden = np.array([reference.encode_design(a) for a in actions])
    return {"max_reward_abs_diff": float(np.max(np.abs(rewards - expected))),
            "max_encode_abs_diff": float(np.max(np.abs(encoded - golden)))}
