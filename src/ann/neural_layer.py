"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from src.ann import Parameter
from src.ann import Module
from src.ann.activations import ReLU
from src.ann.objective_functions import CrossEntropy

class NeuralLayer(Module):
    def __init__(self, in_features: int, out_features: int, init: str = "xavier"):
        super().__init__()
        
        if init == "xavier":
            self.weight = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        elif init == "zeros":
            self.weight = Parameter(np.zeros((in_features, out_features)))
        elif init == "random":
            self.weight = Parameter(np.random.randn(in_features, out_features))
        else:
            raise ValueError(f"Invalid weight initialization method: {init}")

        self.bias = Parameter(np.zeros((1,out_features,)).astype(np.float64))
        
        self.in_features = in_features
        self.out_features = out_features
            
    def forward(self, x: np.ndarray):
        self.input = np.atleast_2d(x).astype(np.float64)
        output = self.input @ self.weight.value + self.bias.value
        return output.astype(np.float64)
    
    def backward(self, grad: np.ndarray):
        grad = np.atleast_2d(grad).astype(np.float64)
        
        grad_x = grad @ self.weight.value.T # dL/dx (to pass to previous layer)
        self.weight.grad = self.input.T @ grad # dL/dw (to update weights)
        self.bias.grad = np.sum(grad, axis=0, keepdims=True) # dL/db (to update biases)
        ## for evaluation
        self.grad_w = self.weight.grad
        self.grad_b = self.bias.grad
        ## for backpropagation
        return grad_x.astype(np.float64)
    
    def __repr__(self):
        return f"NeuralLayer(in_features={self.in_features}, out_features={self.out_features})"
    
    # def grad_w(self):
        # return self.weight.grad
    
    # def grad_b(self):
        # return self.bias.grad


# layer = NeuralLayer(10, 5)
# Relu = ReLU()
# # print(layer.forward(np.random.randn(8, 10)))
# X = np.random.randn(8, 10)
# # print(Relu.forward(layer.forward(X)))
# # print(layer.backward(Relu.backward(np.random.randn(8, 5))))
# # print(layer)

# import torch
# import torch.nn as nn

# # Set the same weights and biases from your custom layer
# torch_layer = nn.Linear(10, 5)

# # Copy weights from your custom layer to torch layer
# torch_layer.weight = nn.Parameter(torch.tensor(layer.weight.value.T, dtype=torch.float64))
# torch_layer.bias = nn.Parameter(torch.tensor(layer.bias.value, dtype=torch.float64))

# # Use the same input X
# X_torch = torch.tensor(X, dtype=torch.float64)

# # Forward pass through torch layer + ReLU
# torch_relu = nn.ReLU()
# torch_output = torch_relu(torch_layer(X_torch))

# # Forward pass through your custom layer + ReLU
# custom_output = Relu.forward(layer.forward(X))

# # Compare results
# # print("Custom output:\n", custom_output)
# # print("\nTorch output:\n", torch_output.detach().numpy())
# print("\nMax difference:", np.max(np.abs(custom_output - torch_output.detach().numpy())))
# print("Outputs match:", np.allclose(custom_output, torch_output.detach().numpy(), atol=1e-6))


# import torch
# import torch.nn.functional as F

# # ── Test data ──────────────────────────────────────────────────────────────────
# np.random.seed(42)
# B, C = 8, 5  # batch size, num classes

# logits_np = np.random.randn(B, C).astype(np.float32)
# y_true_np = np.random.randint(0, C, size=(B,))

# logits_torch = torch.tensor(logits_np, requires_grad=True)
# y_true_torch = torch.tensor(y_true_np, dtype=torch.long)

# # ── Forward pass ───────────────────────────────────────────────────────────────
# ce = CrossEntropy()
# custom_loss = ce.forward(logits_np, y_true_np)
# torch_loss  = F.cross_entropy(logits_torch, y_true_torch)

# print("=== Forward Pass ===")
# print(f"Custom loss : {custom_loss:.6f}")
# print(f"Torch  loss : {torch_loss.item():.6f}")
# print(f"Difference  : {abs(custom_loss - torch_loss.item()):.2e}")
# print(f"Match       : {np.isclose(custom_loss, torch_loss.item(), atol=1e-6)}\n")

# # ── Backward pass ──────────────────────────────────────────────────────────────
# custom_grad = ce.backward()                         # (B, C)
# torch_loss.backward()
# torch_grad  = logits_torch.grad.numpy()             # (B, C)

# print("=== Backward Pass ===")
# print(f"Custom grad (first row): {custom_grad[0]}")
# print(f"Torch  grad (first row): {torch_grad[0]}")
# print(f"Max difference         : {np.max(np.abs(custom_grad - torch_grad)):.2e}")
# print(f"Match                  : {np.allclose(custom_grad, torch_grad, atol=1e-6)}")
