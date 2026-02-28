# ANN Module - Neural Network Implementation

import numpy as np

class Parameter:
    def __init__(self, value):
        if isinstance(value, np.ndarray):
            self.value = value
        else:
            self.value = np.array(value)
        self.grad = np.zeros_like(self.value)


class Module:
    def __init__(self):
        self.weight = None
        self.bias = None

    def forward(self, x: np.ndarray):
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray):
        raise NotImplementedError
    
    def get_params(self):
        if self.weight is None or self.bias is None: ## For activation layers
            raise ValueError("Parameters not initialized")
        return {"weight": self.weight, "biase": self.bias}
    
    def set_params(self, params: dict):
        if self.weight is None or self.bias is None: ## For activation layers
            raise ValueError("Parameters not initialized")
        self.weight = params["weight"]
        self.bias = params["biase"]
    
    def zero_grad(self):
        if self.weight is None or self.bias is None: ## For activation layers
            return
        self.weight.grad = np.zeros_like(self.weight.value)
        self.bias.grad = np.zeros_like(self.bias.value)
