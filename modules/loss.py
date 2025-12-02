import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

class Load_Balancing_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, f, P,):
        return torch.dot(f, P)

class CosineAlignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-8)

    def forward(self, h_sa, h_er):
        h_sa = F.normalize(h_sa, p=2, dim=-1)
        h_er = F.normalize(h_er, p=2, dim=-1)
        sim = self.cos(h_sa, h_er)
        return torch.mean(1.0 - sim)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=None, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - multi-class: (batch_size,)
        """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        probs = F.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        ce_loss = -targets_one_hot * torch.log(probs)

        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss