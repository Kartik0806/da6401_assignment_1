"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from ann import Module
from ann import Parameter

class MSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: np.ndarray, y_true: np.ndarray):
        self.y_true = y_true
        self.logits = logits
        self.prob = softmax(logits)
        self.output = np.mean((y_true - self.prob)**2) 
        return self.output
    
    def backward(self, incoming_grad: np.ndarray = 1):
        dl_dp = 2 * (self.prob - self.y_true) / (self.prob.shape[0] * self.prob.shape[1]) ## (dL/dp)  (B, C)
        dot = np.sum(dl_dp * self.prob, axis=1, keepdims=True)  # (B, 1)
        dL_dz = self.prob * (dl_dp - dot)  # (B, C)
        return incoming_grad * dL_dz ## (1, 1) * (B, C) -> (B, C) 

    def __repr__(self):
        return "MSE()"

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


class CrossEntropy(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: np.ndarray, y_true: np.ndarray):
        
        self.y_true = y_true
        # self.logits = logits
        self.prob = softmax(logits)
        output = np.mean(-np.log(self.prob[np.arange(self.prob.shape[0]), y_true] + 1e-12))
        
        return output 
    
    def backward(self, incoming_grad: np.ndarray = 1):

        local_grad = self.prob.copy()
        local_grad[np.arange(self.prob.shape[0]), self.y_true] -= 1 ## (dL/dz) = (y^ - y) 
        return incoming_grad * local_grad / self.prob.shape[0] ## (1, 1) * (B, 1) -> (B, 1)
    
    def __repr__(self):
        return "CrossEntropy()"

losses = {
    "mse": MSE,
    "cross_entropy": CrossEntropy
}
def get_loss(loss_type: str):
    return losses[loss_type]()
