"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist

def load_dataset(
    name: str,
    seed: int = 42,
    val_size: int = 10000,
    one_hot_labels: bool = False,
):

    name = name.lower().strip()
    if name == "mnist":
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
    elif name == "fashion_mnist":
        (Xtr, ytr), (Xte, yte) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    # Normalize + flatten
    
    Xtr = (Xtr.astype(np.float64) / 255.0).reshape(Xtr.shape[0], -1)
    Xte = (Xte.astype(np.float64) / 255.0).reshape(Xte.shape[0], -1)
    ytr = ytr.astype(int)
    yte = yte.astype(int)

    # Reproducible shuffled train/val split
    rng = np.random.default_rng(seed)
    idx = np.arange(Xtr.shape[0])
    rng.shuffle(idx)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    X_train, y_train = Xtr[train_idx], ytr[train_idx]
    X_val, y_val = Xtr[val_idx], ytr[val_idx]

    if one_hot_labels:
        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        yte = np.eye(10)[yte]

    return X_train, y_train, X_val, y_val, Xte, yte
