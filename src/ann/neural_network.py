"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.activations import get_activation_fn
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer
from utils.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, args):


        self.hidden_sizes = getattr(args, "hidden_sizes", [128, 128])
        
        self.activations = getattr(args, "activation", ["relu", "relu",])
        self.loss_type = getattr(args, "loss", "cross_entropy")
        self.optimizer_type = getattr(args, "optimizer", "sgd")
        self.learning_rate = getattr(args, "learning_rate", 0.01)
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.weight_init = getattr(args, "weight_init", ["xavier", "xavier"])
        self.input_dim = getattr(args, "input_dim", 784)
        self.num_classes = getattr(args, "num_classes", 10)
        self.hidden_sizes = getattr(args, "hidden_sizes", [128, 128])

        acts = getattr(args, "activation", "relu")
        inits = getattr(args, "weight_init", "xavier")


        # If user passes a single string, expand to the number of hidden layers
        if isinstance(acts, str):
            acts = [acts] * len(self.hidden_sizes)
        if isinstance(inits, str):
            inits = [inits] * len(self.hidden_sizes)

        # If user passes a shorter list, extend it
        if len(acts) < len(self.hidden_sizes):
            acts = acts + [acts[-1]] * (len(self.hidden_sizes) - len(acts))
        if len(inits) < len(self.hidden_sizes):
            inits = inits + [inits[-1]] * (len(self.hidden_sizes) - len(inits))

        self.activations = acts
        self.weight_init = inits
        self.build()
        self.first_pass = True
        # raise ValueError(f"Invalid weight initialization method: {self.weight_init}")
    def build(self):

        ## building model
        prev_size = self.input_dim
        self.layers = []
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
        # if self.first_pass:
        #     self.first_pass = False
        #     # self.input_dim = X.shape[1]
        #     # self.build()

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
        self.current_loss = loss
        grad = self.loss.backward()
        # grad_W_list.append(self.layers[-1].grad_w)
        # grad_b_list.append(self.layers[-1].grad_b)
        
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
            if isinstance(layer, NeuralLayer):
                grad_W_list.append(layer.grad_w)
                grad_b_list.append(layer.grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        
        return self.grad_W, self.grad_b


        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        # return self.grad_W, self.grad_b

    def update_weights(self):
        
        self.optimizer.step()


    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32, wandb_run=None):

        self.num_classes = len(np.unique(y_train))
        if wandb_run is not None:
            wandb_run.define_metric("train/step_loss", step_metric="batch_step")
            wandb_run.define_metric("train/epoch_loss", step_metric="epoch")
            wandb_run.define_metric("train/*", step_metric="epoch")
            wandb_run.define_metric("val/*", step_metric="epoch")
            wandb_run.define_metric("weight/*", step_metric="batch_step")
            wandb_run.define_metric("weight/*", step_metric="batch_step")

            print("logging")

        step = 0
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = len(X_train) // batch_size
            
            for i in range(0, len(X_train), batch_size):    

                if wandb_run is not None:
                    wandb_run.log({"weight/weight_1": self.layers[-1].weight.value.flatten()[0], "batch_step": step})
                    wandb_run.log({"weight/weight_2": self.layers[-1].weight.value.flatten()[1], "batch_step": step})
                    wandb_run.log({"weight/weight_3": self.layers[-1].weight.value.flatten()[2], "batch_step": step})
                    wandb_run.log({"weight/weight_4": self.layers[-1].weight.value.flatten()[3], "batch_step": step})
                    wandb_run.log({"weight/weight_5": self.layers[-1].weight.value.flatten()[4], "batch_step": step})

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                for layer in self.layers:
                    layer.zero_grad()
                
                logits = self.forward(X_batch)
                grad_W, grad_b = self.backward(y_batch, logits)
                self.update_weights()
                step += 1
                running_loss += self.current_loss
                
                if wandb_run is not None:
                    wandb_run.log({"train/step_loss": self.current_loss, "batch_step": step})  
                    wandb_run.log({"weight/grad_W": np.linalg.norm(grad_W[-1]), "batch_step": step})
                    wandb_run.log({"weight/grad_b": np.linalg.norm(grad_b[-1]), "batch_step": step})
        

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
        layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                d[f"W{layer_idx}"] = layer.weight.value.copy()
                d[f"b{layer_idx}"] = layer.bias.value.copy()
                layer_idx += 1
        return d

    def set_weights(self, weight_dict):
        layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, NeuralLayer):
                w_key = f"W{layer_idx}"
                b_key = f"b{layer_idx}"
                if w_key in weight_dict:
                    layer.weight.value = weight_dict[w_key].copy()
                if b_key in weight_dict:
                    layer.bias.value = weight_dict[b_key].copy()
                layer_idx += 1
    
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



