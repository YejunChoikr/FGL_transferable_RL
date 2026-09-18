"""Executable numerical contract, not a training runner. NumPy only."""
import numpy as np

ROWS, COLS = 10, 3


def location_vector():
    r = np.arange(1, ROWS + 1, dtype=np.float64)[:, None]
    c = np.arange(1, COLS + 1, dtype=np.float64)[None, :]
    return (np.sqrt(r * r + c * c) / np.sqrt(ROWS**2 + COLS**2)).ravel().astype(np.float32)


def encode_design(actions, assigned=None):
    a = np.asarray(actions, dtype=np.float32)
    if a.shape != (30,) or not np.isfinite(a).all() or (np.abs(a) > 1).any():
        raise ValueError('actions must be a finite length-30 vector in [-1,1]')
    m = np.ones(30, dtype=bool) if assigned is None else np.asarray(assigned, dtype=bool)
    if m.shape != (30,):
        raise ValueError('assigned must have shape (30,)')
    return np.stack((np.where(m, a, 0), np.where(m, location_vector(), 0)), axis=1).ravel().astype(np.float32)


def physical_thickness(actions, t_min, t_max):
    a = np.asarray(actions, dtype=np.float64)
    if not t_min < t_max:
        raise ValueError('invalid bounds')
    return t_min + (a + 1) * (t_max - t_min) / 2


def design_metrics(displacements, actions, goal, C1=0.5, C2=1.0):
    u = np.asarray(displacements, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    if u.shape != (9,) or a.shape != (30,) or not np.isfinite(u).all() or not np.isfinite(a).all():
        raise ValueError('invalid displacement/actions')
    if goal not in range(2, 9) or C2 < 0 or (np.abs(a) > 1).any():
        raise ValueError('invalid goal/coefficient/actions')
    i = goal - 1
    rho = float(np.mean((a + 1) / 2))
    contrast = float(u[i] - (u[i - 1] + u[i + 1]) / 2)
    numerator = float(u[i] - C1 * (u[i - 1] + u[i + 1]))
    denom = 1.0 if C2 == 0 else 0.5 + C2 * rho
    neighbor_mean = float((u[i - 1] + u[i + 1]) / 2)
    return {'reward': numerator / denom, 'contrast_mm': contrast,
            'kappa': float(u[i] / neighbor_mean) if neighbor_mean != 0 else None,
            'normalized_thickness': rho,
            'bilateral_margin_mm': float(min(u[i] - u[i - 1], u[i] - u[i + 1])),
            'peak_success': bool(u[i] > np.max(np.delete(u, i)))}


def early_auc(episodes, rewards):
    e = np.asarray(episodes, dtype=np.int64)
    r = np.asarray(rewards, dtype=np.float64)
    if e.shape != r.shape or not np.isfinite(r).all():
        raise ValueError('invalid curve')
    mask = (e >= 0) & (e <= 300)
    if not np.array_equal(e[mask], np.arange(0, 301, 25)):
        raise ValueError('required observations: 0,25,...,300, exactly once each')
    return float(np.sum(np.diff(e[mask]) * (r[mask][:-1] + r[mask][1:]) / 2) / 300)


def select_checkpoint(episodes, selection_scores):
    """One score per goal per episode; one shared checkpoint across goals."""
    e = np.asarray(episodes, dtype=np.int64)
    s = np.asarray(selection_scores, dtype=np.float64)
    if not np.array_equal(e, np.arange(0, 1501, 25)):
        raise ValueError('required observations: 0,25,...,1500')
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2 or s.shape[0] != len(e) or not np.isfinite(s).all():
        raise ValueError('invalid checkpoint scores')
    k = int(np.argmax(np.mean(s, axis=1)))
    return int(e[k])


def traversal_orders():
    raster = [r * 3 + c for r in range(10) for c in range(3)]
    zigzag = [r * 3 + c for r in range(10) for c in (range(3) if r % 2 == 0 else range(2, -1, -1))]
    spiral = [r * 3 for r in range(10)] + [28, 29] + [r * 3 + 2 for r in range(8, -1, -1)] + [1] + [r * 3 + 1 for r in range(1, 9)]
    return {'rowwise_raster': raster, 'zigzag_inward': zigzag,
            'zigzag_outward': zigzag[::-1], 'spiral_inward': spiral,
            'spiral_outward': spiral[::-1]}


def verify_contract():
    loc = location_vector()
    assert len(loc) == 30 and np.all(loc > 0) and loc[-1] == 1
    a = np.linspace(-1, 1, 30, dtype=np.float32)
    full = encode_design(a)
    assert np.array_equal(full[::2], a) and np.array_equal(full[1::2], loc)
    assert np.array_equal(full.reshape(1, 1, 10, 6).ravel(), full)
    for order in traversal_orders().values():
        assert sorted(order) == list(range(30))
        assigned = np.zeros(30, dtype=bool)
        for i in order:
            assigned[i] = True
            partial = encode_design(a, assigned)
            assert np.count_nonzero(partial[1::2]) == np.count_nonzero(assigned)
        assert np.array_equal(partial, full)
    empty = encode_design(np.zeros(30), np.zeros(30, dtype=bool))
    first = np.zeros(30, dtype=bool); first[0] = True
    filled_zero = encode_design(np.zeros(30), first)
    assert not empty.any() and filled_zero[0] == 0 and filled_zero[1] > 0
    e = np.arange(0, 1501, 25)
    assert early_auc(e, np.ones(len(e))) == 1
    assert np.isclose(early_auc(e, e / 300), 0.5)
    assert select_checkpoint(e, np.ones((61, 5))) == 0
    scores = np.zeros((61, 5)); scores[7, :] = 2; scores[8, 0] = 3
    assert select_checkpoint(e, scores) == 175
    u = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    m = design_metrics(u, np.zeros(30), 5)
    assert m['reward'] == 1 and m['peak_success'] and m['bilateral_margin_mm'] == 1
    assert m['kappa'] == 1.25
    u[0] = 6
    assert not design_metrics(u, np.zeros(30), 5)['peak_success']
    assert design_metrics(u, np.zeros(30), 5)['bilateral_margin_mm'] == 1
    u[0] = 5
    assert not design_metrics(u, np.zeros(30), 5)['peak_success']
    return {'location_shape': [30], 'state_shape': [60], 'all_five_traversals_end_identically': True,
            'zero_action_distinguishable_from_unassigned': True, 'auc_constant_and_linear_tests': True,
            'single_checkpoint_and_earliest_tie_tests': True, 'global_peak_vs_bilateral_test': True}


if __name__ == '__main__':
    import json
    print(json.dumps(verify_contract(), indent=2))
