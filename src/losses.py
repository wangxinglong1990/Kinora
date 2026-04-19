#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class TaskWeightedMSELoss(nn.Module):
    def __init__(
        self,
        km_weight: float = 1.0,
        kcat_weight: float = 1.0,
        loss_type: str = "huber",
        huber_beta: float = 1.0,
    ):
        super().__init__()
        self.km_weight = km_weight
        self.kcat_weight = kcat_weight
        self.loss_type = loss_type
        self.huber_beta = huber_beta

    def forward(self, pred, target):
        if self.loss_type == "mse":
            km_loss = ((pred[:, 0] - target[:, 0]) ** 2).mean()
            kcat_loss = ((pred[:, 1] - target[:, 1]) ** 2).mean()
        else:
            km_loss = nn.functional.smooth_l1_loss(pred[:, 0], target[:, 0], beta=self.huber_beta)
            kcat_loss = nn.functional.smooth_l1_loss(pred[:, 1], target[:, 1], beta=self.huber_beta)
        return self.km_weight * km_loss + self.kcat_weight * kcat_loss

