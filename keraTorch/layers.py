# AUTOGENERATED! DO NOT EDIT! File to edit: Layers.ipynb (unless otherwise specified).

__all__ = ['__inputDimError__', 'Dense', 'Conv2D', 'MaxPool2D', 'Flatten', 'Activation']

# Cell
import numpy as np
import torch.nn as nn
from fastai.vision import *
from fastai import layers

from .activations import *
from functools import partial

# Cell
class __inputDimError__(Exception):
    pass

# Cell
class Dense:
    def __init__(self, units, input_dim=None, activation=None,
                 use_bias=True, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None):
        """
        Linear layer that takes in `input_dim` and converts it to `units` number of dimensions.
        parameters:
        - units: output dimension.
        - input_dim: input dimension.
        - activation (optional): non-linear activation.
        - use_bias (optional): To include bias layer or not (default: True)
        """
        super().__init__()

        self.input_dim = input_dim
        self.activation = get_activation(activation) if activation else None
        if input_dim:
            self.layer = nn.Linear(input_dim, units, bias=use_bias)
        else:
            self.layer = partial(nn.Linear, out_features=units, bias=use_bias)
        # TODO: implement regularizers

    def get_layer(self, input_dim=None):
        if input_dim is None and self.input_dim is None:
            __inputDimError__("Need to specify number of input dimensions in first layer")
        elif input_dim:
            self.input_dim = input_dim
            self.layer = self.layer(in_features=input_dim)
        # else self.layer is already is assigned

        self.output_dim = self.layer.out_features
        layers = [layer for layer in [self.layer, self.activation] if layer]

        return {'output_dim': self.output_dim, 'layers': layers}

# Cell
class Conv2D:
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:int=None,
                 activation:str=None, use_bias:bool=True, input_shape:tuple=None):
        """
        Apply convolution on image using kernel filters.
        parameters:
        - filters: number of kernel filters
        - kernel_size: the width of the (square) kernel
        - strides: number of pixels to skip when sliding kernel (default 1)
        - padding: number of pixels to pad incoming image ` defaults to `ks//2`
        - activation: non-linearity
        - use_bias: bias
        - input_shape: incoming image shape of (#Channels, width, height)
        """
        self.input_shape = input_shape
        if input_shape:
            ni = input_shape[0]
            self.layer = conv2d(ni, filters, kernel_size, strides, padding, use_bias)
        else:
            self.layer = partial(conv2d, nf=filters, ks=kernel_size,
                                 stride=strides, padding=padding, bias=use_bias)
        self.activation = get_activation(activation) if activation else None

    def get_layer(self, input_shape=None):
        if input_shape is None and self.input_shape is None:
            __inputDimError__("Need to specify input shape in first layer")
        elif input_shape:
            self.input_shape = input_shape
            ni = self.input_shape[0]
            self.layer = self.layer(ni=ni)
        # else self.input_shape is already is assigned

        dummy_x = torch.zeros(self.input_shape).unsqueeze(0)
        self.output_shape = self.layer(dummy_x).shape[1:]
        layers = [layer for layer in [self.layer, self.activation] if layer]

        return {'output_dim': self.output_shape, 'layers': layers}

# Cell
class MaxPool2D:
    def __init__(self, kernel_size:int=2, strides:int=None, padding:int=0,
                 input_shape:tuple=None):
        """
        Apply convolution on image using kernel filters.
        parameters:
        - filters: number of kernel filters
        - kernel_size: the width of the (square) kernel
        - strides: number of pixels to skip when sliding kernel (default 1)
        - padding: number of pixels to pad incoming image ` defaults to `ks//2`
        - activation: non-linearity
        - use_bias: bias
        - input_shape: incoming image shape of (#Channels, width, height)
        """
        self.input_shape = input_shape
        self.layer = nn.MaxPool2d(kernel_size, strides, padding)


    def get_layer(self, input_shape=None):
        if input_shape is None and self.input_shape is None:
            __inputDimError__("Need to specify input shape in first layer")
        elif input_shape:
            self.input_shape = input_shape
        # else self.input_shape is already is assigned

        dummy_x = torch.zeros(self.input_shape).unsqueeze(0)
        self.output_shape = self.layer(dummy_x).shape[1:]
        layers = [self.layer]

        return {'output_dim': self.output_shape, 'layers': layers}

# Cell
class Flatten:
    def __init__(self, input_shape=None):
        self.layer = layers.Flatten()
        self.input_shape = input_shape

    def get_layer(self, input_shape=None):
        if input_shape is None and self.input_shape is None:
            __inputDimError__("Need to specify input shape in first layer")
        elif input_shape:
            self.input_shape = input_shape
        # else self.input_shape is already is assigned

        self.output_dim = np.prod(self.input_shape)
        layers = [self.layer]

        return {'output_dim': self.output_dim, 'layers': layers}

# Cell
class Activation:
    def __init__(self, activation, input_shape=None):
        self.layer = get_activation(activation)
        self.input_shape = input_shape
        self.output_dim = input_shape

    def get_layer(self, input_shape=None):
        if input_shape is None and self.input_shape is None:
            __inputDimError__("Need to specify input shape in first layer")
        elif input_shape:
            self.input_shape = input_shape
            self.output_dim = input_shape
        # else self.input_shape is already is assigned

        layers = [self.layer]

        return {'output_dim': self.output_dim, 'layers': layers}