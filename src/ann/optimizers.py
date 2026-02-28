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

class NAG:
    def __init__(self, model, lr: float):
        self.model = model
        self.lr = lr
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    param.value -= self.lr * param.grad

class Momentum:
    def __init__(self, model, lr: float):
        self.model = model
        self.lr = lr
    
    def step(self):
        for layer in self.model.layers:
            if layer.weight is not None:
                params = layer.get_params()
                for name, param in params.items():
                    param.value -= self.lr * param.grad

optims = {
    "sgd": SGD,
    "nag": NAG,
    "momentum": Momentum
}
def get_optimizer(optim_type: str, model, lr: float):
    return optims[optim_type](model, lr)

