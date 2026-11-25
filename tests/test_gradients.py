"""Finite-difference gradient checks for every operation, plus graph behaviour."""

import numpy as np
import pytest

from gradcheck import check_grad
from minigrad import Tensor

TOL = 1e-6
rng = np.random.default_rng(42)


def randn(*shape):
    return rng.standard_normal(shape)


def away_from_zero(*shape):
    """Values with |x| in [0.5, 2], so relu kinks and 1/x blow-ups stay out of reach of eps."""
    return rng.uniform(0.5, 2.0, shape) * rng.choice([-1.0, 1.0], shape)


def positive(*shape):
    return rng.uniform(0.5, 2.0, shape)


# (name, function, inputs)
CASES = [
    # element-wise binary ops, same shape and broadcast
    ("add", lambda a, b: a + b, [randn(2, 3), randn(2, 3)]),
    ("add (3,1)+(1,4)", lambda a, b: a + b, [randn(3, 1), randn(1, 4)]),
    ("add (2,3)+(3,)", lambda a, b: a + b, [randn(2, 3), randn(3)]),
    ("add (2,3)+()", lambda a, b: a + b, [randn(2, 3), randn()]),
    ("sub", lambda a, b: a - b, [randn(2, 3), randn(2, 3)]),
    ("sub (3,1)-(1,4)", lambda a, b: a - b, [randn(3, 1), randn(1, 4)]),
    ("sub (3,)-(2,3)", lambda a, b: a - b, [randn(3), randn(2, 3)]),
    ("mul", lambda a, b: a * b, [randn(2, 3), randn(2, 3)]),
    ("mul (3,1)*(1,4)", lambda a, b: a * b, [randn(3, 1), randn(1, 4)]),
    ("mul (2,3)*(3,)", lambda a, b: a * b, [randn(2, 3), randn(3)]),
    ("mul (4,1,3)*(2,1)", lambda a, b: a * b, [randn(4, 1, 3), randn(2, 1)]),
    ("div", lambda a, b: a / b, [randn(2, 3), away_from_zero(2, 3)]),
    ("div (3,1)/(1,4)", lambda a, b: a / b, [randn(3, 1), away_from_zero(1, 4)]),
    ("div (2,3)/(3,)", lambda a, b: a / b, [randn(2, 3), away_from_zero(3)]),
    ("neg", lambda a: -a, [randn(2, 3)]),
    # reversed operators with Python numbers
    ("radd", lambda a: 2 + a, [randn(2, 3)]),
    ("rsub", lambda a: 1 - a, [randn(2, 3)]),
    ("rmul", lambda a: 2 * a, [randn(2, 3)]),
    ("rtruediv", lambda a: 1 / a, [away_from_zero(2, 3)]),
    # matmul
    ("matmul", lambda a, b: a @ b, [randn(3, 4), randn(4, 2)]),
    ("matmul batched", lambda a, b: a @ b, [randn(5, 3, 4), randn(4, 2)]),
    # reductions
    ("sum all", lambda a: a.sum(), [randn(2, 3, 4)]),
    ("sum axis=0", lambda a: a.sum(axis=0), [randn(2, 3, 4)]),
    ("sum axis=-1", lambda a: a.sum(axis=-1), [randn(2, 3, 4)]),
    ("sum axis=(0,2)", lambda a: a.sum(axis=(0, 2)), [randn(2, 3, 4)]),
    ("sum axis=1 keepdims", lambda a: a.sum(axis=1, keepdims=True), [randn(2, 3, 4)]),
    ("sum all keepdims", lambda a: a.sum(keepdims=True), [randn(2, 3)]),
    ("mean all", lambda a: a.mean(), [randn(2, 3, 4)]),
    ("mean axis=1", lambda a: a.mean(axis=1), [randn(2, 3, 4)]),
    ("mean axis=(0,-1) keepdims", lambda a: a.mean(axis=(0, -1), keepdims=True), [randn(2, 3, 4)]),
    # shape ops
    ("reshape", lambda a: a.reshape(4, 6), [randn(2, 3, 4)]),
    ("reshape tuple -1", lambda a: a.reshape((3, -1)), [randn(2, 3, 4)]),
    ("transpose", lambda a: a.transpose(), [randn(2, 3, 4)]),
    ("transpose axes", lambda a: a.transpose(1, 2, 0), [randn(2, 3, 4)]),
    ("T", lambda a: a.T, [randn(2, 3)]),
    # element-wise functions
    ("relu", lambda a: a.relu(), [away_from_zero(3, 4)]),
    ("exp", lambda a: a.exp(), [randn(3, 4)]),
    ("log", lambda a: a.log(), [positive(3, 4)]),
    # a small composite: one linear layer + relu + mean
    ("linear+relu+mean", lambda x, W, b: (x @ W + b).relu().mean(), [randn(5, 4), randn(4, 3), randn(3)]),
]


@pytest.mark.parametrize("name, f, inputs", CASES, ids=[c[0] for c in CASES])
def test_gradient_matches_finite_differences(name, f, inputs):
    assert check_grad(f, *inputs) < TOL


def test_tensor_used_twice():
    # y = x*x + x, so dy/dx = 2x + 1. Both uses of x must add their contributions.
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True, dtype=np.float64)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)
    assert check_grad(lambda x: x * x + x, x.data) < TOL


def test_shared_intermediate_needs_topological_order():
    # a feeds c both directly and through b, so a's gradient is only complete once
    # b has been processed. A wrong order would push a partial gradient into x.
    def f(x):
        a = x.exp()
        b = a * x
        return a + b

    x = np.array([0.5, -1.0, 2.0])
    t = Tensor(x, requires_grad=True, dtype=np.float64)
    f(t).sum().backward()
    np.testing.assert_allclose(t.grad, np.exp(x) * (x + 2))  # d/dx [e^x (1 + x)]
    assert check_grad(f, x) < TOL


def test_backward_twice_accumulates_into_leaves_and_zero_grad_resets():
    x = Tensor([1.0, 2.0], requires_grad=True, dtype=np.float64)
    y = (x * 3).sum()
    y.backward()
    y.backward()
    np.testing.assert_allclose(x.grad, [6.0, 6.0])  # 3 + 3, not more
    x.zero_grad()
    assert x.grad is None
    y.backward()
    np.testing.assert_allclose(x.grad, [3.0, 3.0])


def test_constants_get_no_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    c = Tensor([3.0, 4.0])
    (x * c).sum().backward()
    assert c.grad is None
    with pytest.raises(RuntimeError):
        c.sum().backward()


def test_dtypes():
    assert Tensor([1, 2]).dtype == np.float32
    x32 = Tensor([1.0, 2.0], requires_grad=True)
    y32 = (2 * x32 - 1).exp().mean()
    assert y32.dtype == np.float32
    y32.backward()
    assert x32.grad.dtype == np.float32

    x64 = Tensor([1.0, 2.0], requires_grad=True, dtype=np.float64)
    y64 = (1 / x64 + 3).log().sum()
    assert y64.dtype == np.float64
    y64.backward()
    assert x64.grad.dtype == np.float64
