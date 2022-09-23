""" Cosine Scheduler
Cosine schedule with warmup.
Copyright 2021 Ross Wightman
"""
import math
import torch

from timm.scheduler.scheduler import Scheduler


class CosineScheduler(Scheduler):
    """
    Cosine decay with warmup.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Modified from timm's implementation.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_name: str,
                 t_max: int,
                 value_min: float = 0.,
                 warmup_t=0,
                 const_t=0,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field=param_name, initialize=initialize)

        assert t_max > 0
        assert value_min >= 0
        assert warmup_t >= 0
        assert const_t >= 0

        self.cosine_t = t_max - warmup_t - const_t
        self.value_min = value_min
        self.warmup_t = warmup_t
        self.const_t = const_t

        if self.warmup_t:
            self.warmup_steps = [(v - value_min) / self.warmup_t for v in self.base_values]
            super().update_groups(self.value_min)
        else:
            self.warmup_steps = []

    def _get_value(self, t):
        if t < self.warmup_t:
            values = [self.value_min + t * s for s in self.warmup_steps]
        elif t < self.warmup_t + self.const_t:
            values = self.base_values
        else:
            t = t - self.warmup_t - self.const_t

            value_max_values = [v for v in self.base_values]

            values = [
                self.value_min + 0.5 * (value_max - self.value_min) * (1 + math.cos(math.pi * t / self.cosine_t))
                for value_max in value_max_values
            ]

        return values

    def get_epoch_values(self, epoch: int):
        return self._get_value(epoch)
