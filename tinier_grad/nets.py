import numpy as np 

import tinier_grad.tensor 

class Module(): 
    '''
    Base module class for all .nn classes 
    '''
    def __init__(self): 
        pass

    def modules(self): 
        '''
        Returns an iterator over all the modules in the network
        '''
        pass

    def train(self, mode: bool = True): 
        '''
        Sets the module in training mode
        '''
        pass


    def eval(self): 
        '''
        Sets the module in evaluation mode
        '''
        pass

    def requires_grad(self, requires_grad: bool = True): 
        '''
        Determines if autograd should record the changes on parameters in the module
        '''
        pass

    def zero_grad(self): 
        '''
        Set gradients of parameters to zero to start next step of optimizer
        '''
        pass


class ReLU(): 
    def __init__(self): 
        pass
    # pass in function from tensor.py here

class Linear(Module): 
    def __init(self, in_channels, out_channels): 
        pass
    # pass in linear functiion from tensor.py here

class Conv2D(Module): 
    def __init__(self): 
        pass

    # pass in linear functiion from tensor.py here
