"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
from src.ann import Parameter
# from src.ann.neural_network import NeuralNetwork
## Access each layer's parameters and grads using get_params() and update them
class SGD:
    
    def __init__(self, model, lr: float):
        self.model = model
        self.lr = lr
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    # print(name, param.value.shape, param.grad.shape)
                    param.value -= self.lr * param.grad

import numpy as np

class Momentum:
    
    def __init__(self, model, lr: float, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    key = id(param)
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(param.value)
                    self.velocity[key] = self.momentum * self.velocity[key] - self.lr * param.grad
                    param.value += self.velocity[key]


class NAG:
    
    def __init__(self, model, lr: float, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    key = id(param)
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(param.value)
                    v_prev = self.velocity[key].copy()
                    self.velocity[key] = self.momentum * self.velocity[key] - self.lr * param.grad
                    param.value += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[key]


class RMSprop:
    
    def __init__(self, model, lr: float, beta: float = 0.9, eps: float = 1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    key = id(param)
                    if key not in self.cache:
                        self.cache[key] = np.zeros_like(param.value)
                    self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (param.grad ** 2)
                    param.value -= self.lr * param.grad / (np.sqrt(self.cache[key]) + self.eps)

optims = {
    "sgd": SGD,
    "nag": NAG,
    "momentum": Momentum,
    "rmsprop": RMSprop
}
def get_optimizer(optim_type: str, model, lr: float):
    return optims[optim_type](model, lr)

