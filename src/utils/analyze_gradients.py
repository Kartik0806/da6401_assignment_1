from src.ann.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from src.ann.neural_layer import NeuralLayer
import numpy as np
import wandb

def analyze_gradients(model:NeuralNetwork, wandb_run = None):

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

def analyze_weights(model:NeuralNetwork, layer_ids:list[int], wandb_run = None):

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

def analyze_activations(model:NeuralNetwork, X: np.ndarray, wandb_run = None):

    # columns = ["Layer Number", "Activations"]
    # table = wandb.Table(columns = columns)
    activations = []

    for layer in model.layers:
        X = layer.forward(X)

        ## only append activations for neural layers
        if not isinstance(layer, NeuralLayer):
            activations.append(X)
    
    n = len(activations)

    cols = 3
    rows = (n+cols -1) // cols

    fig = plt.figure(figsize=(15, 4 * rows))

    for i in range(len(activations)):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.hist(activations[i].flatten())
        ax.set_title(f"Activation distribution for layer {i}")

    if wandb_run is not None:
        wandb_run.log({"activations": wandb.Image(fig)})
    
    plt.tight_layout()

    plt.show()
    
