"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np
from src.ann import Module
from src.ann import Parameter

## grad is incoming gradient from the next layer, i.e. dL/dz
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    def backward(self, grad):
        local_grad = np.where(self.input > 0, 1, 0)
        return grad * local_grad
    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        local_grad = self.output * (1 - self.output)
        return grad * local_grad
    def __repr__(self):
        return "Sigmoid()"

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
    
    def forward(self,x):
        self.input = x
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad):
        local_grad = 1 - self.output**2
        return grad * local_grad
    
    def __repr__(self):
        return "Tanh()"