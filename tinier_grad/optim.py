import numpy as np 
from tensor import Tensor 
from types import List 


class Optimizer(): 
    def __init__(self, params: List[Tensor]): 
        for i in params: 
            if i.requires_grad is None:
                i.requires_grad == True

        self.params = params

    def zero_grad(self): 
        for param in self.params: 
            param.grad = None

class SGD(Optimizer):
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self): 
        pass

class MomSGD(Optimizer): 
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self): 
        pass


class Nestrov(Optimizer): 
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass


class Adagrad(Optimizer): 
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass

class Adadelta(Optimizer):
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass

class RMSprop(Optimizer): 
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass

class Adam(Optimizer):
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass
    
def AdaMax(Optimizer):
    def __init__(self, params: List[Tensor]): 
        self.params = params

    def step(self):
        pass





