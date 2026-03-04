"""
2.1 Data Exploration and Class Distribution
Logs a W&B Table with 5 sample images from each of the 10 Fashion-MNIST classes.
"""

import numpy as np
import wandb
from src.utils.data_loader import load_dataset

CLASS_NAMES = [
    "0",  
    "1",  
    "2", 
    "3", 
    "4", 
    "5", 
    "6", 
    "7", 
    "8", 
    "9", 
]
# CLASS_NAMES = [
#     "T-shirt/top",  # 0
#     "Trouser",      # 1
#     "Pullover",     # 2
#     "Dress",        # 3
#     "Coat",         # 4
#     "Sandal",       # 5
#     "Shirt",        # 6
#     "Sneaker",      # 7
#     "Bag",          # 8
#     "Ankle boot",   # 9
# ]

SAMPLES_PER_CLASS = 5


def main(seed: int = 42):
    dataset = "mnist"
    (X_train, y_train, X_val, y_val, X_test, y_test) = load_dataset(dataset, flatten=False)  

    rng = np.random.default_rng(seed)

    run = wandb.init(
        project="wandb_report",
        name="class-distribution",
        tags=["eda", "class-distribution"],
    )

    columns = ["class_id", "class_name"] + [f"sample_{i+1}" for i in range(SAMPLES_PER_CLASS)]
    table = wandb.Table(columns=columns)

    for class_id in range(10):
        class_name = CLASS_NAMES[class_id]

        indices = np.where(y_train == class_id)[0]
        chosen = rng.choice(indices, size=SAMPLES_PER_CLASS, replace=False)

        images = [
            wandb.Image(X_train[idx], caption=f"{class_name} #{i+1}")
            for i, idx in enumerate(chosen)
        ]

        table.add_data(class_id, class_name, *images)

    run.log({"class_distribution": table})
    run.finish()


if __name__ == "__main__":
    main()