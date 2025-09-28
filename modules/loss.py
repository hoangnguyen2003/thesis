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