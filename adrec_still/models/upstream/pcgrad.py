"""PCGrad multi-objective gradient merge (ported from the fork's ``src/pcgrad.py``).

The fork wraps *every* optimizer in this class. With ``pcgrad=False`` (the
upstream default) the projection is skipped but the merge still applies:
shared-parameter gradients are the **mean** of the per-objective gradients,
non-shared gradients their sum — so ADRec's CE + scaled-MSE combination trains
through the same merge as upstream, whether or not the projection is enabled.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch


class PCGrad:
    def __init__(self, optimizer, *, pcgrad: bool = False, reduction: str = "mean") -> None:
        self._optim, self._reduction = optimizer, reduction
        self.pcgrad = pcgrad

    @property
    def optimizer(self):
        return self._optim

    @property
    def optim(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives) -> None:
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = copy.deepcopy(grads)
        if self.pcgrad:
            for g_i in pc_grad:
                random.shuffle(grads)
                for g_j in grads:
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0:
                        g_i -= g_i_g_j * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == "mean":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            raise ValueError("invalid reduction method")
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads) -> None:
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
        return grads, shapes, has_grads

    @staticmethod
    def _unflatten_grad(grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    @staticmethod
    def _flatten_grad(grads):
        return torch.cat([g.flatten() for g in grads])

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
