"""Compare MiniGrad's gradients with PyTorch's on random inputs (skipped without torch)."""

import numpy as np
import pytest

from minigrad import Tensor

torch = pytest.importorskip("torch")

# (name, minigrad function, torch function, input shapes, kind of input values)
CASES = [
    ("add bcast", lambda a, b: a + b, lambda a, b: a + b, [(3, 1), (1, 4)], "any"),
    ("sub bcast", lambda a, b: a - b, lambda a, b: a - b, [(2, 3), (3,)], "any"),
    ("mul bcast", lambda a, b: a * b, lambda a, b: a * b, [(4, 1, 3), (2, 1)], "any"),
    ("div bcast", lambda a, b: a / b, lambda a, b: a / b, [(2, 3), (3,)], "nonzero"),
    ("neg", lambda a: -a, lambda a: -a, [(2, 3)], "any"),
    ("reversed ops", lambda a: 1 - 2 * a + 3 / a, lambda a: 1 - 2 * a + 3 / a, [(2, 3)], "nonzero"),
    ("matmul", lambda a, b: a @ b, lambda a, b: a @ b, [(3, 4), (4, 2)], "any"),
    ("matmul batched", lambda a, b: a @ b, lambda a, b: a @ b, [(5, 3, 4), (4, 2)], "any"),
    ("sum", lambda a: a.sum(axis=(0, 2)), lambda a: a.sum(dim=(0, 2)), [(2, 3, 4)], "any"),
    ("sum keepdims", lambda a: a.sum(axis=1, keepdims=True), lambda a: a.sum(dim=1, keepdim=True), [(2, 3, 4)], "any"),
    ("mean", lambda a: a.mean(axis=-1), lambda a: a.mean(dim=-1), [(2, 3, 4)], "any"),
    ("mean all", lambda a: a.mean(), lambda a: a.mean(), [(2, 3, 4)], "any"),
    ("reshape", lambda a: a.reshape(4, 6), lambda a: a.reshape(4, 6), [(2, 3, 4)], "any"),
    ("transpose", lambda a: a.transpose(1, 2, 0), lambda a: a.permute(1, 2, 0), [(2, 3, 4)], "any"),
    ("relu", lambda a: a.relu(), lambda a: a.relu(), [(3, 4)], "any"),
    ("exp", lambda a: a.exp(), lambda a: a.exp(), [(3, 4)], "any"),
    ("log", lambda a: a.log(), lambda a: a.log(), [(3, 4)], "positive"),
    (
        "two-layer mlp",
        lambda x, W1, b1, W2: ((x @ W1 + b1).relu() @ W2).exp().mean(),
        lambda x, W1, b1, W2: ((x @ W1 + b1).relu() @ W2).exp().mean(),
        [(8, 5), (5, 6), (6,), (6, 3)],
        "any",
    ),
]


def make_inputs(shapes, kind, seed):
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]
    if kind == "positive":
        arrays = [np.abs(a) + 0.1 for a in arrays]
    elif kind == "nonzero":
        arrays = [np.where(a >= 0, 1, -1) * (np.abs(a) + 0.5) for a in arrays]
    return arrays


def grads(mg_f, torch_f, arrays, dtype):
    mg_in = [Tensor(a, requires_grad=True, dtype=dtype) for a in arrays]
    th_dtype = torch.float64 if dtype == np.float64 else torch.float32
    th_in = [torch.tensor(a, dtype=th_dtype, requires_grad=True) for a in arrays]

    mg_out, th_out = mg_f(*mg_in), torch_f(*th_in)
    w = np.random.default_rng(1).standard_normal(mg_out.shape)  # seed gradient
    mg_out.backward(w)
    th_out.backward(torch.tensor(w, dtype=th_dtype))
    return mg_out, th_out, mg_in, th_in


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name, mg_f, torch_f, shapes, kind", CASES, ids=[c[0] for c in CASES])
def test_matches_torch_float64(name, mg_f, torch_f, shapes, kind, seed):
    arrays = make_inputs(shapes, kind, seed)
    mg_out, th_out, mg_in, th_in = grads(mg_f, torch_f, arrays, np.float64)
    np.testing.assert_allclose(mg_out.data, th_out.detach().numpy(), rtol=1e-12, atol=1e-12)
    for m, t in zip(mg_in, th_in):
        np.testing.assert_allclose(m.grad, t.grad.numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name, mg_f, torch_f, shapes, kind", CASES, ids=[c[0] for c in CASES])
def test_matches_torch_float32(name, mg_f, torch_f, shapes, kind):
    arrays = make_inputs(shapes, kind, seed=3)
    mg_out, th_out, mg_in, th_in = grads(mg_f, torch_f, arrays, np.float32)
    np.testing.assert_allclose(mg_out.data, th_out.detach().numpy(), rtol=1e-5, atol=1e-6)
    for m, t in zip(mg_in, th_in):
        assert m.grad.dtype == np.float32
        np.testing.assert_allclose(m.grad, t.grad.numpy(), rtol=1e-4, atol=1e-5)
