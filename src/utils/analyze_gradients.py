from src.ann.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from src.ann.neural_layer import NeuralLayer
import numpy as np

def analyze_gradients(model:NeuralNetwork):

    grad_W = model.grad_W
    grad_b = model.grad_b
    print(grad_W.shape)
    for i in range(len(grad_W)):
        plt.hist(grad_W[i][0].flatten(),)
        plt.title(f"Gradient distribution for layer {i}")
        plt.show()
        plt.hist(grad_b[i][0].flatten(),)
        plt.title(f"Gradient distribution for layer {i}")
        plt.show()

def analyze_weights(model:NeuralNetwork, layer_ids: list[int]):

    for layer_id in layer_ids:
        layer = model.layers[layer_id]
        if layer.weight is not None:
            plt.hist(layer.weight.value.flatten(), bins=50)
            plt.title(f"Weight distribution for layer {layer_id}")
            plt.show()
        if layer.bias is not None:
            plt.hist(layer.bias.value.flatten(), bins=50)
            plt.title(f"Bias distribution for layer {layer_id}")
            plt.show()

def analyze_activations(model:NeuralNetwork, X: np.ndarray):
    activations = []

    for layer in model.layers:
        X = layer.forward(X)

        ## only append activations for neural layers
        if not isinstance(layer, NeuralLayer):
            activations.append(X)
    
    for i in range(len(activations)):
        plt.hist(activations[i].flatten())
        plt.title(f"Activation distribution for layer {i}")
        plt.show()