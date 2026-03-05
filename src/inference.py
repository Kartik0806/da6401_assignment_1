"""
Inference Script
Evaluate trained models on test sets
"""

import argparse

from sklearn import metrics
from utils.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
import numpy as np 

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist"],
        required=False,
        help="Dataset to use",
        default="mnist"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, required=False, help="Number of training epochs", default=10
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, required=False, help="Mini-batch size", default=64
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        choices=["cross_entropy", "mean_squared_error", "mse"],
        default="cross_entropy",
        help="Loss function",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        default="sgd",
        help="Optimizer type",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        help="Learning rate for optimizer",
        default=0.1
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for L2 regularization",
    )
    parser.add_argument(
        "-nhl",
        "--num_layers",
        type=int,
        required=False,
        help="Number of hidden layers",
        default=2
    )
    parser.add_argument(
        "-sz",
        "--hidden_sizes",
        nargs="+",
        type=int,                       
        required=False,
        help="Number of neurons in each hidden layer (one value per layer)",
        default=[128, 128]
    )
    parser.add_argument(
        "-a",
        "--activation",
        nargs="+",
        type=str,
        choices=["relu", "sigmoid", "tanh"],
        default=["relu", "relu"],
        help=(
            "Activation function(s) for hidden layers. "
            "Pass one value to broadcast to all layers, "
            "or one per hidden layer e.g. -a relu sigmoid tanh"
        ),
    )
    parser.add_argument(
        "-w_i",
        "--weight_init",
        nargs="+",
        type=str,
        choices=["random", "xavier", "zeros"],
        default=["xavier", "xavier"],
        help=(
            "Weight initialization method(s) for hidden layers. "
            "Pass one value to broadcast to all layers, "
            "or one per hidden layer e.g. -w_i xavier random xavier"
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name",

    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for W&B",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model.npy",
        help="Relative path to save trained model weights (.npy)",
    )

    return parser.parse_args()


def load_model(model_path,):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()

    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    loss = model.loss.forward(logits, y_test)
    acc = accuracy_score(y_test, logits)
    precision = precision_score(y_test, logits)
    recall = recall_score(y_test, logits)
    f1 = f1_score(y_test, logits)
    return {"loss": loss, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    _, _, _, _, X_test, y_test = load_dataset(args.dataset)
    
    # global model
    model = NeuralNetwork(args)
    weights = load_model(args.model_save_path)
    model.set_weights(weights)
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation complete!")
    print(metrics)
    # print(weights)
    return metrics

if __name__ == '__main__':
    main()
