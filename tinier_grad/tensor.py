import numpy as np

class Tensor:
    def __init__(self, data, _children=(), requires_grad=True):

        self.data = data
        if isinstance(self.data, list):
            self.data = np.array(self.data)

        self.requires_grad = requires_grad
        self._prev = set(_children)
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def ones(cls, *shape): 
        return cls(np.ones(shape, dtype=np.float32))

    @classmethod
    def empty(cls, *shape):
        return cls(np.empty(shape, dtype=np.float32))

    @classmethod
    def randn(cls, *shape):
        return cls(np.randn(shape, dtype=np.float32))

    @classmethod
    def zeros(cls, *shape): 
        return cls(np.zeros(shape, dtype=np.float32))

    @classmethod
    def zeros_like(cls, tensor): 
        return cls.zeros(np.zeros_like(*tensor.shape))

    @classmethod
    def eye(cls, *shape):
        return cls(np.eye(shape, dtype=np.float32))

    def backward(self):
        if self.requires_grad:

            # topological order all of the children in the graph
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            # go one variable at a time and apply the chain rule to get its gradient
            self.grad = 1
            for v in reversed(topo):
                v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other), requires_grad=self.requires_grad or other.requires_grad)
        inputs = [self, other]
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            
        def _backward():
            if self.requires_grad: 
                self.grad += other.data * out.grad
            if other.requires_grad: 
                other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def matmul(self,other):
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward(): 
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.T)
            if other.requires_grad: 
                other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Tensor(self.data ** other, requires_grad=self.requires_grad)
        def _backward(): 
            if self.requires_grad: 
                self.grad += out.grad * out.data*self.data ** (out.data -1)
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad)
        def _backward(): 
            if self.requires_grad: 
                self.grad += out.data * (1 - out.data) * out.grad
        return out

    def relu(self): 
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        def _backward(): 
            if self.requires_grad: 
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    # TODO: maybe add tensor like properties like in pytorch?
    @classmethod
    def randn(cls, *shape): 
        return cls(np.randn(*shape))

    

if __name__ =='__main__': 
    x = Tensor([1, 2])
    y = Tensor([3, 4])
    z = y * x
    z = z + 2
    z.backward() 
    print(x.grad)
