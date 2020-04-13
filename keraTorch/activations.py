# AUTOGENERATED! DO NOT EDIT! File to edit: Activations.ipynb (unless otherwise specified).

__all__ = ['relu', 'ReLU', 'LeakyReLU', 'leaky_relu', 'Mish', 'mish', 'sigmoid', 'softmax', 'get_activation',
           '__activations__']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell
def relu(inplace=True):
    return nn.ReLU(inplace=inplace)

ReLU = Relu = relu

# Cell
def LeakyReLU(inplace=True, negative_slope=0.01):
    return nn.LeakyReLU(negative_slope, inplace=inplace)

leaky_relu = leakyrelu = LeakyReLU

# Cell
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

mish = Mish

# Cell
sigmoid = nn.Sigmoid

# Cell
softmax = nn.Softmax(dim=-1)

# Cell
__activations__ = {
    'relu': relu(),
    'leaky_relu': leaky_relu(),
    'mish': mish(),
    'sigmoid': sigmoid(),
    'softmax': softmax
}

def get_activation(activation):
    return __activations__[activation.lower()]