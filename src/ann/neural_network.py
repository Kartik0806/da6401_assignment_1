"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from src.ann.activations import get_activation_fn
from src.ann.neural_layer import NeuralLayer
from src.ann.objective_functions import get_loss
from src.ann.optimizers import get_optimizer
from src.utils.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, args: dict):
        self.input_dim = args["input_dim"]
        self.hidden_sizes = args["hidden_sizes"]
        self.activations = args["activations"]
        self.num_classes = args["num_classes"]
        self.loss_type = args["loss_type"]
        self.optimizer_type = args["optimizer_type"]
        self.learning_rate = args["learning_rate"]
        self.weight_decay = args["weight_decay"]
        self.weight_init = args["weight_init"]
        self.layers = []

        self.build()

    def build(self):

        ## building model
        prev_size = self.input_dim
        for i, (hhz, act) in enumerate(zip(self.hidden_sizes, self.activations)):
            self.layers.append(NeuralLayer(prev_size, hhz, init = self.weight_init[i]))
            self.layers.append(get_activation_fn(act))
            prev_size = hhz
        self.layers.append(NeuralLayer(prev_size, self.num_classes))

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
        grad = self.loss.backward()
        # grad_W_list.append(self.layers[-1].grad_w)
        # grad_b_list.append(self.layers[-1].grad_b)
        
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
            if isinstance(layer, NeuralLayer):
                grad_W_list.append(layer.grad_w)
                grad_b_list.append(layer.grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes

        if len(grad_W_list) < 50:
            self.grad_W = np.empty(len(grad_W_list), dtype=object)
            self.grad_b = np.empty(len(grad_b_list), dtype=object)
            for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
                self.grad_W[i] = gw
                self.grad_b[i] = gb
        
        return loss, self.grad_W, self.grad_b


        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        # return self.grad_W, self.grad_b

    def update_weights(self):
        
        self.optimizer.step()


    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32, wandb_run=None):

        if wandb_run is not None:
            wandb_run.define_metric("train/step_loss", step_metric="batch_step")
            wandb_run.define_metric("train/epoch_loss", step_metric="epoch")
            wandb_run.define_metric("train/*", step_metric="epoch")
            wandb_run.define_metric("val/*", step_metric="epoch")
            print("logging")

        step = 0
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = len(X_train) // batch_size
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                for layer in self.layers:
                    layer.zero_grad()
                
                logits = self.forward(X_batch)
                loss, grad_W, grad_b = self.backward(y_batch, logits)
                self.update_weights()
                step += 1
                running_loss += loss
                
                if wandb_run is not None:
                    wandb_run.log({"train/step_loss": loss, "batch_step": step})  
                    wandb_run.log({"train/grad_W": np.linalg.norm(grad_W[-1]), "batch_step": step})
                    wandb_run.log({"train/grad_b": np.linalg.norm(grad_b[-1]), "batch_step": step})
            

            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val)
            if wandb_run is not None:
                wandb_run.log({
                    "train/epoch_loss": running_loss / num_batches,
                    **{f"train/{k}": v for k, v in train_metrics.items() if k != "loss"},
                    "val/epoch_loss": val_metrics["loss"],
                    **{f"val/{k}": v for k, v in val_metrics.items() if k != "loss"},
                    "epoch": epoch
                }) 
                    
            print(f"Train loss: {running_loss / num_batches}")
            print(f"Val loss: {val_metrics['loss']}")

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss = self.loss.forward(logits, y)
        acc = accuracy_score(y, logits)
        precision = precision_score(y, logits)
        recall = recall_score(y, logits)
        f1 = f1_score(y, logits)
        return {"loss": loss, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NeuralLayer):
                d[f"W{i}"] = layer.weight.value.copy()
                d[f"b{i}"] = layer.bias.value.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NeuralLayer):
                w_key = f"W{i}"
                b_key = f"b{i}"
                if w_key in weight_dict:
                    layer.weight.value = weight_dict[w_key].copy()
                if b_key in weight_dict:
                    layer.bias.value = weight_dict[b_key].copy()
    
    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])


# nn = NeuralNetwork(input_dim=784)
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")

# nn.train(X_train, y_train, epochs=10, batch_size=64)

# print(nn.evaluate(X_train, y_train))
# print(nn.evaluate(X_val, y_val))
# print(nn.evaluate(X_test, y_test))


# X = np.random.randn(8, 16)
# y = np.random.randint(0, 10, size=(8,))
# # for layer in nn.layers:
# #     print(layer)

# # logits = nn.forward(X)
# # print(nn.backward(y, logits))
# # nn.update_weights()
# X_train = np.random.randn(256, 16)
# y_train = np.random.randint(0, 10, size=(256,))

# nn.train(X_train, y_train, epochs=10, batch_size=32)



