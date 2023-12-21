# -*- coding: utf-8 -*-

from functools import lru_cache
from math import prod
import re
from typing import MutableMapping, MutableSequence, Sequence

import torch
import torch.nn as nn


class ModelParallelMixin:

    def parallel(
        self,
        mp_rule: MutableMapping = {},
    ):
        for n, m in self.named_modules():
            for r, dev in mp_rule.items():
                if re.fullmatch(r, n) is not None:
                    print(f"{n} match rule {r}, will be put on device {dev}")
                    m.to(dev)
                    self.apply_forward_hook(m, pre=True, post=False)
                    break

        cpu_parameters = set()
        for n, p in self.named_parameters():
            if p.is_cuda:
                continue

            if n.endswith("weight") or n.endswith("bias"):
                n = n.rsplit(".", 1)[0]

            cpu_parameters.add(n)    

        for cpu_m in cpu_parameters:
            print(f"[WARNING] {cpu_m} is not put on any cuda devices")

        return self

    def apply_forward_hook(self, module, *, pre=False, post=True):
        module.forward = self._forward_hook(module, module.forward, pre=pre, post=post)

    def _forward_hook(
        self,
        module,
        forward,
        *,
        pre=True,
        post=False,
    ):
        device = next(module.parameters()).device

        def wrapper(*args, **kwargs):
            if pre:
                args = self._to_hook(args, device)
                kwargs = self._to_hook(kwargs, device)

            result = forward(*args, **kwargs)

            if post:
                result = self._to_hook(result, device)
            return result

        return wrapper

    def _to_hook(self, x, d):
        if isinstance(x, torch.Tensor):
            return x.to(d)
        elif isinstance(x, MutableMapping):
            for k, v in x.items():
                x[k] = self._to_hook(v, d)
            return x
        elif isinstance(x, MutableSequence):
            for idx, v in enumerate(x):
                x[idx] = self._to_hook(v, d)
            return x
        elif isinstance(x, Sequence):
            return [self._to_hook(v, d) for v in x]
        else:
            return x

    @lru_cache
    def params_num(self, module: nn.Module):
        return sum([prod(p.shape) for p in module.parameters()])
