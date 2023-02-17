import numpy as np 
from types import List

class Conv2d(): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.padding
        self.bias = bias

    def __call__(self): 
        pass

class Linear(): 
    def __init__(self, in_channels, out_channels, weight, bias = True): 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = weight
        self.bias = bias

    def __call__(self): 
        pass

class BatchNorm(): 
    def __init__(self): 
        pass

class LayerNorm(): 
    def __init__(self): 
        pass
