import argparse
import os
import json
import numpy as np
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from utils.analyze_gradients import analyze_gradients, analyze_activations
from utils.metrics import get_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import dotenv
# dotenv.load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def parse_arguments():
    """
    Parse command-line arguments.

    Mandatory arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - num_layers: Number of hidden layers
    - hidden_size: List of hidden layer sizes (one per hidden layer)
    - activation: Activation function per hidden layer — either one value (broadcast)
                  or one per layer e.g. -a relu sigmoid tanh
    - loss: Loss function ('cross_entropy', 'mean_squared_error', 'mse')
    - weight_init: Weight init per hidden layer — either one value (broadcast)
                   or one per layer e.g. -w_i xavier random xavier
    - weight_decay: L2 regularization strength
    - wandb_project: W&B project name
    - model_save_path: Relative path to save trained model
    """
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist"],
        required=True,
        help="Dataset to use",
        default="mnist"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, required=True, help="Number of training epochs", default=10
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, required=True, help="Mini-batch size", default=64
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
        required=True,
        help="Learning rate for optimizer",
        default=0.01
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for L2 regularization",
    )
    parser.add_argument(
        "-nhl",
        "--num_layers",
        type=int,
        required=True,
        help="Number of hidden layers",
        default=1
    )
    parser.add_argument(
        "-sz",
        "--hidden_sizes",
        nargs="+",
        type=int,
        required=True,
        help="Number of neurons in each hidden layer (one value per layer)",
        default=[128]
    )
    parser.add_argument(
        "-a",
        "--activation",
        nargs="+",
        type=str,
        choices=["relu", "sigmoid", "tanh"],
        default=["relu"],
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
        default=["xavier"],
        help=(
            "Weight initialization method(s) for hidden layers. "
            "Pass one value to broadcast to all layers, "
            "or one per hidden layer e.g. -w_i xavier random xavier"
        ),
    )
    parser.add_argument(
        "-w_p",
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
        default="model.npy",
        help="Relative path to save trained model weights (.npy)",
    )


    args = parser.parse_args()


    # if len(args.hidden_size) != args.num_layers: 

    #     raise ValueError("Length of --hidden_size must equal --num_layers")

    # if len(args.activation) == 1: 
    #     args.activation = args.activation * args.num_layers
    # elif len(args.activation) != args.num_layers: 
    #     raise ValueError( 
    #         f"--activation must be 1 value (broadcast) or {args.num_layers} values "
    #         f"(one per hidden layer), got {len(args.activation)}"
    #     ) 

    # if len(args.weight_init) == 1:
    #     args.weight_init = args.weight_init * args.num_layers

    # elif len(args.weight_init) != args.num_layers:
    #     raise ValueError(
    #         f"--weight_init must be 1 value (broadcast) or {args.num_layers} values "
    #         f"(one per hidden layer), got {len(args.weight_init)}"
    #     )

    return args


def main():
    """
    Main training function.
    """
    args = parse_arguments() 
    args.input_dim = 784
    args.num_classes = 10
    model = NeuralNetwork(args)
    # print(args)
    # args.update({"input_dim": 784, "num_classes": 10})
    # model = NeuralNetwork(args)
    # print(model)
    # return
    
 
    one_hot = args.loss != "cross_entropy"

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.dataset, one_hot_labels=one_hot
    ) 

    wandb_run = None 
    import wandb

    if WANDB_API_KEY is not None:
        wandb.login(key=WANDB_API_KEY)

    # if args.wandb_project:
    #     print(args.wandb_project)
    #     print(args.run_name)
    #     wandb_run = None
        # wandb_run = wandb.init(project=args["wandb_project"], config=args, name=args["run_name"])

    # model = NeuralNetwork(args)

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=wandb_run,
    )

    val_metrics = model.evaluate(X_val, y_val)
    train_metrics = model.evaluate(X_train, y_train)
    print("Training metrics:", train_metrics)
    test_metrics = model.evaluate(X_test, y_test)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    if wandb_run is not None:
        wandb_run.log(
            {
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
            }
        )
    print("Training complete!")
    # print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    # print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    weights = model.get_weights()       
    args_dict = vars(args)
    import json

    with open("best_config.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    np.save(args.model_save_path, weights, allow_pickle=True)

    # model = NeuralNetwork(args)
    # loaded_weights = np.load(args.model_save_path, allow_pickle=True).item()
    # model.set_weights(loaded_weights)

    # loaded_metrics = model.evaluate(X_test, y_test)
    # print("Loaded model metrics:", loaded_metrics)

    # analyze_gradients(model)
    # analyze_weights(model, [0, 1, 2])
    # analyze_activations(model, X_test, wandb_run=wandb_run)

    ## to get confusion matrix and log to wandb
    # fig, cnf_matrix = get_confusion_matrix(y_test, model.forward(X_test))

    # wandb_run.log({"confusion_matrix": wandb.Image(fig)})
    # print(cnf_matrix)
    # wandb_run.finish()

if __name__ == "__main__":
    main()