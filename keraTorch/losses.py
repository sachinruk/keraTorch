# AUTOGENERATED! DO NOT EDIT! File to edit: losses.ipynb (unless otherwise specified).

__all__ = ['mse_loss', 'mae_loss', 'binary_cross_entropy', 'ce_loss', 'ce4softmax']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell
mse_loss = mean_squared_loss = mse = F.mse_loss

# Cell
mae_loss = mean_absolute_error = mae = F.l1_loss

# Cell
binary_cross_entropy = bce_loss = bce = F.binary_cross_entropy

# Cell
ce_loss = F.cross_entropy

# Cell
def ce4softmax(input, target):
    eps = 1e-9
    logp = torch.log(input + eps)
    return ce_loss(logp, target)