"""Tensor with reverse-mode automatic differentiation."""

import numpy as np


class Tensor:
    """A NumPy array that records how it was computed, so gradients can flow back.

    data          the values (np.ndarray, float32 by default, float64 allowed)
    grad          d(output)/d(self) after backward(), same shape as data, else None
    requires_grad whether gradients should be computed for this tensor
    _parents      the tensors this one was computed from
    _backward     closure that pushes self.grad into the parents' grads
    """

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
