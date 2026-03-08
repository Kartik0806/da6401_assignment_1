import os
import argparse
import wandb
from types import SimpleNamespace

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

SWEEP_CONFIG = {
    "method": "bayes",         
    "metric": {
        "name": "test/f1",
        "goal": "maximize",
    },

    "parameters": {

        "optimizer": {
            "values": ["sgd", "momentum", "nag", "rmsprop", ]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "weight_decay": {
            "values": [0.0, 0.1, 1e-4, 5e-4, 1e-3]
        },
        "num_layers": {
            "values": [1, 2, 3, 4]
        },
        "hidden_size": {        
            "values": [ 256, 128, 64, 32 ]
        },
        "activation": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "batch_size": {
            "values": [64, 128, 256]
        },
        "epochs": {
            "values": [15, 10]
        },
        "weight_init": {
            "values": [ "xavier"]
        },
        "loss": {
            "values": ["cross_entropy", "mean_squared_error"]
        },
    },
}

def train():

    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_dataset

    run = wandb.init()          
    cfg = dict(run.config)

    n = cfg["num_layers"]
    cfg["hidden_sizes"]  = [cfg["hidden_size"]] * n

    cfg.update({"input_dim": 784, "num_classes": 10, "dataset": DATASET})

    one_hot = cfg["loss"] != "cross_entropy"
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        DATASET, one_hot_labels=one_hot
    )
    cfg["analyze_gradients"] = False
    cfg["analyze_activations"] = False
    cfg["analyze_weights"] = False
    cfg["analyze_confusion_matrix"] = False
    cfg = SimpleNamespace(**cfg)
    model = NeuralNetwork(cfg)
    model.train(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=run,
    )

    val_metrics = model.evaluate(X_val, y_val)
    train_metrics = model.evaluate(X_train, y_train)
    print("Training metrics:", train_metrics)
    test_metrics = model.evaluate(X_test, y_test)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    if run is not None:
        run.log(
            {
                "test/accuracy": test_metrics["accuracy"],
                "test/f1": test_metrics["f1"],
                "train/accuracy": train_metrics["accuracy"],
                "train/f1": train_metrics["f1"],
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1"],
            }
        )
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--wandb_project", type=str, default="sweep_test_")
    parser.add_argument("--count",         type=int, default=300,
                        help="Total number of sweep runs (default: 300)")
    parser.add_argument(
        "--analyze_gradients",
        default=False,
        help="Whether to analyze gradients after training",
    )
    parser.add_argument(
        "--analyze_activations",
        default=False,
        help="Whether to analyze activations after training",
    )
    parser.add_argument(
        "--analyze_weights",
        default=False,
        help="Whether to analyze weights after training",
    )
    parser.add_argument(
        "--analyze_confusion_matrix",
        default=False,
        help="Whether to analyze confusion matrix after training",
    )
    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset

    wandb.login(key=WANDB_API_KEY)

    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project=args.wandb_project,
    )
    print(f"Sweep created: {sweep_id}")
    print(f"Launching agent for {args.count} runs …") 
    wandb.agent(sweep_id, function=train, count=args.count)