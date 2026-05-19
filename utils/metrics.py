import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def _to_numpy_int(arr):
    """Convert labels/preds to 1D int numpy array."""
    out = np.asarray(arr)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out.astype(np.int64)


def _softmax_numpy(logits):
    """Stable softmax for numpy arrays."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _weighted_ovr_auc(probs, labels, num_classes):
    """
    Weighted one-vs-rest AUC implemented in numpy.
    Falls back to 0.0 if a class has only one label state in OVR targets.
    """
    labels = _to_numpy_int(labels)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"probs shape must be [N, {num_classes}], got {probs.shape}")

    n = labels.shape[0]
    class_counts = np.bincount(labels, minlength=num_classes)
    weights = class_counts / max(n, 1)

    auc_sum = 0.0
    weight_sum = 0.0

    for k in range(num_classes):
        y_true = (labels == k).astype(np.int32)
        if y_true.min() == y_true.max():
            continue

        scores = probs[:, k]
        order = np.argsort(-scores)
        y_sorted = y_true[order]

        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        pos = tp[-1]
        neg = fp[-1]

        if pos == 0 or neg == 0:
            continue

        tpr = np.concatenate(([0.0], tp / pos, [1.0]))
        fpr = np.concatenate(([0.0], fp / neg, [1.0]))
        auc_k = np.trapz(tpr, fpr)

        wk = weights[k]
        auc_sum += wk * auc_k
        weight_sum += wk

    if weight_sum == 0.0:
        return 0.0
    return float(auc_sum / weight_sum)


def classification_metrics(logits, preds, labels, num_classes):
    """
    Generic multiclass metrics for classification tasks.

    Returns a dict with:
    - acc
    - f1_macro
    - f1_weighted
    - auc (weighted OVR AUC)
    """

    labels = _to_numpy_int(labels)
    preds = _to_numpy_int(preds)

    probs = _softmax_numpy(logits)
    auc_score = _weighted_ovr_auc(probs, labels, num_classes)

    return {
        'acc': float(accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'auc': auc_score,
    }


def distance_weighted_accuracy(y_true, y_pred, alpha=0.5):
    """
    Distance Weighted Accuracy (DW-Acc):
        mean(exp(-alpha * |y - y_hat|))
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)
    dist = np.abs(y_true - y_pred)
    return float(np.mean(np.exp(-float(alpha) * dist)))


def exponential_distance_score(y_true, y_pred, alpha=1.0):
    """
    Exponential Distance Score / F-Acc for ordinal credit risk:
        F-Acc = mean(exp(-alpha * |y - y_hat|))

    Recommended for credit rating where long-distance errors should be
    penalized much more than short-distance errors.
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)
    dist = np.abs(y_true - y_pred)
    return float(np.mean(np.exp(-float(alpha) * dist)))


def credit_distance_score(y_true, y_pred, num_classes, mode='linear', alpha=0.5):
    """
    Credit Distance Score (CDS) for ordinal credit ratings.

    mode='linear':
        CDS = mean(1 - |y - y_hat| / (K - 1))
    mode='exp':
        CDS = mean(exp(-alpha * |y - y_hat|))
    """
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)
    dist = np.abs(y_true - y_pred).astype(np.float64)

    if mode == 'linear':
        denom = max(int(num_classes) - 1, 1)
        score = 1.0 - (dist / denom)
        return float(np.mean(score))

    if mode == 'exp':
        return float(np.mean(np.exp(-float(alpha) * dist)))

    raise ValueError(f"Unsupported mode: {mode}. Use 'linear' or 'exp'.")


def credit_rating_metrics(logits, preds, labels, num_classes):
    """
    Full metric suite for credit rating classification (ordinal-aware).

    Includes standard classification metrics and distance-aware metrics:
    - acc, f1_macro, f1_weighted, auc
    - eda_acc (exponential distance accuracy)
    """
    base = classification_metrics(logits, preds, labels, num_classes)
    eda_acc = exponential_distance_score(labels, preds, alpha=1.0)

    base['eda_acc'] = eda_acc
    return base


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
