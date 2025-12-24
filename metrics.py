import numpy as np
from sklearn.metrics import f1_score


def confidence_weighted_f1(y_true, y_pred, confidence):
    weighted_tp = np.sum(confidence * (y_pred == 1) * (y_true == 1))
    weighted_fp = np.sum(confidence * (y_pred == 1) * (y_true == 0))
    weighted_fn = np.sum(confidence * (y_pred == 0) * (y_true == 1))

    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def macro_conf_weighted_f1(y_true, y_pred, y_proba, classes):
    f1_per_class = [
        confidence_weighted_f1((y_true == i), (y_pred == i), y_proba[:, i])
        for i in range(len(classes))
    ]
    return float(np.mean(f1_per_class))


def official_global_f1(y_true, y_pred, y_proba, classes):
    n_classes = len(classes)
    f1_per_class = np.array(
        [
            confidence_weighted_f1((y_true == i), (y_pred == i), y_proba[:, i])
            for i in range(n_classes)
        ]
    )
    supports = np.bincount(y_true, minlength=n_classes).astype(float)

    macro_f1 = f1_per_class.mean()
    weighted_f1 = np.average(f1_per_class, weights=supports) if supports.sum() > 0 else 0.0
    return float(max(macro_f1, weighted_f1))


def compute_metrics(y_true, y_proba, classes):
    y_pred = y_proba.argmax(axis=1)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "official_global_f1": official_global_f1(y_true, y_pred, y_proba, classes),
    }
