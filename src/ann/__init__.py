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
        self.weights = None 
        self.biases = None  

    def forward(self, x: np.ndarray):
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray):
        raise NotImplementedError
    
    def get_params(self):
        if self.weights is None or self.biases is None: ## For activation layers
            raise ValueError("Parameters not initialized")
        return {"weights": self.weights, "biases": self.biases}
    
    def set_params(self, params: dict):
        if self.weights is None or self.biases is None: ## For activation layers
            raise ValueError("Parameters not initialized")
        self.weights = params["weights"]
        self.biases = params["biases"]
    
    def zero_grad(self):
        if self.weights is None or self.biases is None: ## For activation layers
            return
        self.weights.grad = np.zeros_like(self.weights.value)
        self.biases.grad = np.zeros_like(self.biases.value)
