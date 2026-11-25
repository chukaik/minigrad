"""Tensor with reverse-mode automatic differentiation."""

import numpy as np


def unbroadcast(grad, shape):
    """Sum grad over the dimensions that broadcasting added or stretched, back to `shape`.

    Broadcasting copies an input along some axes, so in the backward pass every
    copy's gradient has to be added back into the one original value.
    """
    while grad.ndim > len(shape):  # axes broadcasting prepended
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):  # axes stretched from size 1
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array that records how it was computed, so gradients can flow back.

    data          the values (np.ndarray, float32 by default, float64 allowed)
    grad          d(output)/d(self) after backward(), same shape as data, else None
    requires_grad whether gradients should be computed for this tensor
    _parents      the tensors this one was computed from
    _backward     closure that pushes self.grad into the parents' grads
    """

    __array_ufunc__ = None  # make `ndarray * Tensor` call Tensor.__rmul__ instead of NumPy

    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.asarray(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = ()
        self._backward = lambda: None

    # ---- small conveniences ----

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # ---- graph building helpers ----

    def _make(self, data, parents):
        """Wrap an op's result; it needs a gradient if any parent does."""
        out = Tensor(data, dtype=data.dtype)
        out.requires_grad = any(p.requires_grad for p in parents)
        if out.requires_grad:
            out._parents = parents
        return out

    def _accumulate(self, g):
        """Add g into self.grad (a tensor used twice receives the sum of both)."""
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    # ---- reverse pass ----

    def backward(self, grad=None):
        """Fill .grad of every tensor in the graph that leads to self.

        Leaf tensors accumulate across calls (use zero_grad() between steps);
        intermediate tensors are reset, so calling backward() twice is safe.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")

        # Topological order: every node appears after all the nodes it was computed from.
        # Iterative DFS so deep graphs don't hit Python's recursion limit.
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        for node in order:
            if node._parents:
                node.grad = None

        seed = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=self.dtype)
        self._accumulate(seed)
        for node in reversed(order):  # output first, leaves last
            if node._parents and node.grad is not None:
                node._backward()

    def zero_grad(self):
        """Forget the stored gradient."""
        self.grad = None

    # ---- operations ----
    # In each backward closure, g = out.grad is d(loss)/d(out).

    def _lift(self, other):
        """Turn a number or array into a constant Tensor with self's dtype."""
        return other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)

    def __add__(self, other):
        """a + b.  grad: da = g, db = g (each unbroadcast)."""
        other = self._lift(other)
        out = self._make(self.data + other.data, (self, other))

        def _backward():
            self._accumulate(unbroadcast(out.grad, self.shape))
            other._accumulate(unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __sub__(self, other):
        """a - b.  grad: da = g, db = -g (each unbroadcast)."""
        other = self._lift(other)
        out = self._make(self.data - other.data, (self, other))

        def _backward():
            self._accumulate(unbroadcast(out.grad, self.shape))
            other._accumulate(unbroadcast(-out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        """a * b.  grad: da = g * b, db = g * a (each unbroadcast)."""
        other = self._lift(other)
        out = self._make(self.data * other.data, (self, other))

        def _backward():
            self._accumulate(unbroadcast(out.grad * other.data, self.shape))
            other._accumulate(unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """a / b.  grad: da = g / b, db = -g * a / b**2 (each unbroadcast)."""
        other = self._lift(other)
        out = self._make(self.data / other.data, (self, other))

        def _backward():
            self._accumulate(unbroadcast(out.grad / other.data, self.shape))
            other._accumulate(
                unbroadcast(-out.grad * self.data / other.data**2, other.shape)
            )

        out._backward = _backward
        return out

    def __neg__(self):
        """-a.  grad: da = -g."""
        out = self._make(-self.data, (self,))

        def _backward():
            self._accumulate(-out.grad)

        out._backward = _backward
        return out

    # Reversed operators: 2 * t, 1 - t, ... (Python calls these when the left side isn't a Tensor).
    def __radd__(self, other):
        return self._lift(other) + self

    def __rsub__(self, other):
        return self._lift(other) - self

    def __rmul__(self, other):
        return self._lift(other) * self

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __matmul__(self, other):
        """a @ b (2-D or batched).  grad: da = g @ b.T, db = a.T @ g."""
        other = self._lift(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul needs tensors with at least 2 dimensions")
        out = self._make(self.data @ other.data, (self, other))

        def _backward():
            g = out.grad
            self._accumulate(unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            other._accumulate(unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum over axis.  grad: g copied back along the summed axes."""
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the summed axes back as size 1
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean over axis.  grad: g / n copied back, n = number of averaged elements."""
        out = self.sum(axis=axis, keepdims=keepdims)
        return out / (self.data.size // max(out.data.size, 1))

    def reshape(self, *shape):
        """Same values, new shape.  grad: g reshaped back to the input shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,))

        def _backward():
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """Permute axes (reverse them if none given).  grad: g permuted back by the inverse."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.ndim)))
        out = self._make(self.data.transpose(axes), (self,))

        def _backward():
            self._accumulate(out.grad.transpose(np.argsort(axes)))

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def relu(self):
        """max(a, 0).  grad: da = g where a > 0, else 0."""
        out = self._make(np.maximum(self.data, 0), (self,))

        def _backward():
            self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def exp(self):
        """e**a.  grad: da = g * e**a."""
        out = self._make(np.exp(self.data), (self,))

        def _backward():
            self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self):
        """Natural log.  grad: da = g / a."""
        out = self._make(np.log(self.data), (self,))

        def _backward():
            self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out
