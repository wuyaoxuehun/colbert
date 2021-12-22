#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn.functional as F
import torch
from torch.autograd import Function


class PoincareDistance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2)) \
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None

def hyperdist(u, v):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-5
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)

def l2norm(x):
    return F.normalize(x, dim=-1, p=2)


def exp_map(x):
    length = torch.sqrt((x ** 2).sum(-1))
    # input(length)
    # print(length.size(), x.size())
    # input()
    return torch.tanh(length)[:, None] * x / (length[:, None])

def test_PD():
    a = torch.randn(2, 2, 4).requires_grad_(True)
    b = torch.randn(3, 3, 4).requires_grad_(True)
    a1 = a.view(-1, 4).repeat_interleave(9, dim=0)
    b1 = b.view(-1, 4).repeat((4, 1))
    print(a1.size(), b1.size())
    a_, b_ = exp_map(a1), exp_map(b1)
    dist = PoincareDistance.apply(a_, b_, 1e-5)
    # dist = torch.cosh(dist) ** 2
    dist = (dist ** 2) / 2
    print(dist)
    dist = dist.view(2, 2, 3, 3).max(-1)[0].sum(1)

    label = torch.randn(2, 3)
    loss = F.mse_loss(dist, label)
    print(dist)
    loss.backward()
    print(a.grad)

    pass

def test():
    a = torch.tensor([1, 0])
    b = torch.tensor([0, 200])
    print(PoincareDistance.apply(a, b, 1e-5))


if __name__ == '__main__':
    test_PD()
    # test()