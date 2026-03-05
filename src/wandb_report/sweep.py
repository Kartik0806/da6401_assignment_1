"""
Hyperparameter Sweep using W&B
Performs a Bayesian sweep with 100+ runs over key hyperparameters.
Run with: python sweep.py --dataset fashion_mnist --wandb_project <your-project>
"""

import os
import argparse
import wandb
import dotenv
dotenv.load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")


SWEEP_CONFIG = {
    "method": "bayes",         
    "metric": {
        "name": "val/accuracy",
        "goal": "maximize",
    },
    "early_terminate": {      
        "type": "hyperband",
        "min_iter": 3,
    },
    "parameters": {

        "optimizer_type": {
            "values": ["sgd", "momentum", "nag", "rmsprop", ]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e-1,
        },
        "weight_decay": {
            "values": [0.0, 1e-4, 5e-4, 1e-3]
        },
        "num_layers": {
            "values": [1, 2, 3, 4]
        },
        "hidden_size": {        
            "values": [32, 64, 128, 256, 512]
        },
        "activation": {
            "values": ["relu", "sigmoid", "tanh"]
        },
        "batch_size": {
            "values": [16, 32, 64, 128, 256]
        },
        "epochs": {
            "value": 10          
        },
        "weight_init": {
            "values": [ "xavier"]
        },
        "loss_type": {
            "values": ["cross_entropy", "mse"]
        },
    },
}

def train():

    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_dataset

    run = wandb.init()          
    cfg = dict(run.config)

    n = cfg["num_layers"]
    cfg["hidden_sizes"]  = [cfg.pop("hidden_size")] * n
    cfg["activations"]   = [cfg.pop("activation")]  * n
    cfg["weight_init"]   = [cfg.pop("weight_init")] * n
    cfg.update({"input_dim": 784, "num_classes": 10, "dataset": DATASET})

    one_hot = cfg["loss_type"] != "cross_entropy"
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        DATASET, one_hot_labels=False
    )

    model = NeuralNetwork(cfg)
    model.train(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        X_val=X_val,
        y_val=y_val,
        wandb_run=run,
    )

    val_metrics  = model.evaluate(X_val,   y_val)
    test_metrics = model.evaluate(X_test, y_test)

    run.log({
        "val/loss":      val_metrics["loss"],
        "val/accuracy":  val_metrics["accuracy"],
        "test/loss":     test_metrics["loss"],
        "test/accuracy": test_metrics["accuracy"],
    })
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--count",         type=int, default=100,
                        help="Total number of sweep runs (default: 100)")
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