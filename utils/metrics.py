import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    pred_ids = np.argmax(predictions, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, pred_ids),
        "f1": f1_score(labels, pred_ids, average="weighted"),
        "precision": precision_score(labels, pred_ids, average="weighted", zero_division=0),
        "recall": recall_score(labels, pred_ids, average="weighted", zero_division=0)
    }
