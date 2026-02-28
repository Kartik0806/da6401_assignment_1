"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from src.ann.activations import get_activation_fn
from src.ann.neural_layer import NeuralLayer
from src.ann.objective_functions import get_loss
from src.ann.optimizers import get_optimizer

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, 
        input_dim: int = 16,
        hidden_sizes: list[int] = [8, 16, 8],
        activations: list[str] = ["relu", "relu", "relu"], 
        output_size: int = 10,
        loss_type: str = "cross_entropy",
        optimizer_type: str= "sgd",
        learning_rate: float= 0.01,
        num_classes: int= 10,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activations = activations
        self.output_size = output_size
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.layers = []

        self.build()

    def build(self):

        ## building model
        prev_size = self.input_dim
        for hhz, act in zip(self.hidden_sizes, self.activations):
            self.layers.append(NeuralLayer(prev_size, hhz))
            self.layers.append(get_activation_fn(act))
            prev_size = hhz
        self.layers.append(NeuralLayer(prev_size, self.output_size))

        ## getting loss
        self.loss = get_loss(self.loss_type)

        ## getting optimizer
        self.optimizer = get_optimizer(self.optimizer_type, model = self, lr = self.learning_rate)

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        loss = self.loss.forward(y_pred, y_true)
        print(loss)

        grad = self.loss.backward()
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        
        return loss
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):

        self.optimizer.step()
        pass

    def train(self, X_train, y_train, epochs=1, batch_size=32):

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                logits = self.forward(X_batch)
                loss = self.backward(y_batch, logits)
                self.update_weights()

    def evaluate(self, X, y):
        pass

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()


nn = NeuralNetwork()
X = np.random.randn(8, 16)
y = np.random.randint(0, 10, size=(8,))
# for layer in nn.layers:
#     print(layer)

# logits = nn.forward(X)
# print(nn.backward(y, logits))
# nn.update_weights()
X_train = np.random.randn(256, 16)
y_train = np.random.randint(0, 10, size=(256,))

nn.train(X_train, y_train, epochs=10, batch_size=32)

