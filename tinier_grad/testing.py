import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._op = None
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self):
        if self.requires_grad:
            self._op.backward(self)

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._op = AddBackward()
        out._op.inputs = [self, other]
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._op = MulBackward()
        out._op.inputs = [self, other]
        return out

    def __pow__(self, other):
        out_data = self.data ** other
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._op = PowBackward()
        out._op.inputs = [self, out]
        if self.requires_grad:
            out._op.output_grad = np.ones_like(out_data)
        return out


class AddBackward:
    def backward(self, output):
        for inp in self.inputs:
            if inp.requires_grad:
                inp.grad += output.grad

class MulBackward:
    def backward(self, output):
        for inp in self.inputs:
            if inp.requires_grad:
                inp.grad += output.grad * (self.inputs[1].data if inp is self.inputs[0] else self.inputs[0].data)

class PowBackward:
    def backward(self, output):
        self.inputs[0].grad += output.grad * self.inputs[1].data * self.inputs[0].data ** (self.inputs[1].data - 1)


x = Tensor(np.array([2, 3]), requires_grad=True)
y = x ** 2
y.backward()

print(x.grad)  # prints [4 6]

