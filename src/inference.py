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

    parser.add_argument("--model_path", type=str, default="best_model.npy", help="Relative path to load trained model weights (.npy)")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-nhl", "--num_layers", type=int,  default=2)
    parser.add_argument("-sz", "--hidden_size", type=int,  default=128)
    parser.add_argument(
        "-a", "--activation",
        type=str,
        choices=["relu", "sigmoid", "tanh"],
        default="relu"
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        choices=["random", "xavier", "zeros"],
        default="xavier"
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
    if np.max(X_test) > 1.0:
        X_test = X_test.astype(np.float64) / 255.0
    
    logits = model.forward(X_test)
    loss = model.loss.forward(logits, y_test)
    acc = accuracy_score(y_test, logits)
    precision = precision_score(y_test, logits)
    recall = recall_score(y_test, logits)
    # f1 = f1_score(y_test, logits)
    f1 = metrics.f1_score(y_pred=np.argmax(logits, axis=1), y_true=y_test, average='micro')

    # acc = metrics.accuracy_score(y_test, np.argmax(logits, axis=1))
    # precision = metrics.precision_score(y_test, np.argmax(logits, axis=1), average='weighted')
    # recall = metrics.recall_score(y_test, np.argmax(logits, axis=1), average='weighted')
    # f1 = metrics.f1_score(y_test, np.argmax(logits, axis=1), average='weighted')
    return {"loss": loss, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    import json
    with open("best_config.json", "r") as f:
        best_config = json.load(f)
    
    args = argparse.Namespace(**best_config)

    one_hot = args.loss != "cross_entropy"

    _, _, _, _, X_test, y_test = load_dataset(
        args.dataset, one_hot_labels=one_hot
    ) 
    
    model = NeuralNetwork(args)
    weights = load_model(args.model_save_path)   
    model.set_weights(weights)
    result = evaluate_model(model, X_test, y_test)
    print("Evaluation complete!")
    print(result)
    return result

if __name__ == '__main__':
    main()
