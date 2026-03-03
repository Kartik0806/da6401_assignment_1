import numpy as np

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
    
    
    