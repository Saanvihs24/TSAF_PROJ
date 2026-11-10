import numpy as np
import pandas as pd

PEAK_HOURS = (18, 19, 20, 21)     # target hours, 6 to 9 pm
THRESHOLD_P90 = 65.0              # alert threshold (training 90th percentile)
THRESHOLD_P95 = 73.0              # "severe" threshold (training 95th percentile)


def mase_scale(train_target, season=168):
    """Mean absolute error of the 'same hour last week' rule on the training target."""
    t = np.asarray(train_target, dtype=float)
    return float(np.mean(np.abs(t[season:] - t[:-season])))


def forecast_metrics(y_true, y_pred, target_time, scale):
    """Accuracy metrics. target_time = the hour being predicted (origin + horizon)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    peak = pd.DatetimeIndex(target_time).hour.isin(PEAK_HOURS)

    mae = np.mean(np.abs(err))
    return {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(y_true))),
        "MASE": float(mae / scale),
        "Bias": float(np.mean(err)),                       # positive = over-predicting
        "Peak_MAE": float(np.mean(np.abs(err[peak]))) if peak.any() else np.nan,
    }


def alert_metrics(y_true, y_pred, threshold):
    """Treat 'index at or above the threshold' as an event and score the predicted alerts."""
    actual = np.asarray(y_true, dtype=float) >= threshold
    alert = np.asarray(y_pred, dtype=float) >= threshold
    tp = int(np.sum(actual & alert))
    fp = int(np.sum(~actual & alert))
    fn = int(np.sum(actual & ~alert))
    tn = int(np.sum(~actual & ~alert))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    def f_beta(b):
        if not (tp + fp) or not (tp + fn) or (precision + recall) == 0:
            return np.nan
        return (1 + b ** 2) * precision * recall / (b ** 2 * precision + recall)

    return {
        "Precision": precision, "Recall": recall,
        "F1": f_beta(1), "F2": f_beta(2),
        "FalseAlarmRate": fp / (fp + tn) if (fp + tn) else np.nan,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def evaluate(y_true, y_pred, target_time, scale, threshold=THRESHOLD_P90):
    """Accuracy and alert metrics in one dictionary."""
    return {**forecast_metrics(y_true, y_pred, target_time, scale),
            **alert_metrics(y_true, y_pred, threshold)}