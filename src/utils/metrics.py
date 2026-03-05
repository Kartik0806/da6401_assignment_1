import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(y_true, logits):
    y_pred = np.argmax(logits, axis=1)

    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    num_classes = np.max(y_true) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(im)

    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, conf_matrix[i, j],
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black"
            )
    plt.tight_layout()
    plt.show()

    return fig, conf_matrix

def accuracy_score(y_true, logits):
    y_pred = np.argmax(logits, axis=1)

    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(y_true == y_pred)

def precision_score(y_true, logits):

    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    num_classes = np.max(y_true) + 1
    precision = np.zeros(num_classes)

    y_pred = np.argmax(logits, axis=1)

    for i in range(num_classes):
        tp = np.sum((y_true == y_pred) & (y_pred == i))
        fp = np.sum((y_true != y_pred) & (y_pred == i))
        precision[i] = tp / (tp + fp)
    
    return np.mean(precision)

def recall_score(y_true, logits):
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    num_classes = np.max(y_true) + 1
    recall = np.zeros(num_classes)

    y_pred = np.argmax(logits, axis=1)

    for i in range(num_classes):
        tp = np.sum((y_true == y_pred) & (y_true == i))
        fn = np.sum((y_true != y_pred) & (y_true == i))
        recall[i] = tp / (tp + fn)
    
    return np.mean(recall)

def f1_score(y_true, logits):
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    precision = precision_score(y_true, logits)
    recall = recall_score(y_true, logits)

    f1 = 2 * (precision * recall) / (precision + recall)
    return np.mean(f1)
    
    
    