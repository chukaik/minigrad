"""Compare MiniGrad's gradients with central finite differences, in float64."""

import numpy as np

from minigrad import Tensor


def check_grad(f, *arrays, eps=1e-6, seed=0):
    """Return the worst relative error between backward() and finite differences.

    f takes Tensors and returns a Tensor of any shape. It is reduced to a scalar
    with fixed random weights w, so every output element gets a different seed
    gradient (a plain .sum() would hide some mistakes).
    """
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    w = np.random.default_rng(seed).standard_normal(
        f(*[Tensor(a, dtype=np.float64) for a in arrays]).shape
    )

    def scalar(*xs):
        return float((f(*[Tensor(x, dtype=np.float64) for x in xs]).data * w).sum())

    inputs = [Tensor(a, requires_grad=True, dtype=np.float64) for a in arrays]
    out = f(*inputs)
    out.backward(w)

    worst = 0.0
    for i, a in enumerate(arrays):
        numeric = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            plus, minus = [x.copy() for x in arrays], [x.copy() for x in arrays]
            plus[i][idx] += eps
            minus[i][idx] -= eps
            numeric[idx] = (scalar(*plus) - scalar(*minus)) / (2 * eps)
        analytic = inputs[i].grad
        assert analytic is not None and analytic.shape == a.shape
        assert analytic.dtype == np.float64
        denom = max(np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12)
        worst = max(worst, np.linalg.norm(analytic - numeric) / denom)
    return worst
