"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from src.ann import Parameter
from src.ann import Module
from src.ann.activations import ReLU

class NeuralLayer(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        self.weight = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.zeros((out_features,)))
        
        self.in_features = in_features
        self.out_features = out_features
            
    def forward(self, x: np.ndarray):
        self.input = np.atleast_2d(x)
        output = self.input @ self.weight.value + self.bias.value
        return output
    
    def backward(self, grad: np.ndarray):
        grad = np.atleast_2d(grad)
        
        grad_x = grad @ self.weight.value.T # dL/dx (to pass to previous layer)
        self.weight.grad = self.input.T @ grad # dL/dw (to update weights)
        self.bias.grad = np.sum(grad, axis=0, keepdims=True) # dL/db (to update biases)
        ## for evaluation
        self.grad_w = self.weight.grad
        self.grad_b = self.bias.grad
        ## for backpropagation
        return grad_x
    
    def __repr__(self):
        return f"NeuralLayer(in_features={self.in_features}, out_features={self.out_features})"
    
    # def grad_w(self):
        # return self.weight.grad
    
    # def grad_b(self):
        # return self.bias.grad


# layer = NeuralLayer(10, 5)
# Relu = ReLU()
# # print(layer.forward(np.random.randn(8, 10)))
# print(Relu.forward(layer.forward(np.random.randn(8, 10))))
# print(layer.backward(Relu.backward(np.random.randn(8, 5))))
# print(layer)