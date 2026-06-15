import numpy as np
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, roc_curve


metric_rules = {"auroc": "max",
                'f1': 'max',
                'auprc': 'max',
                'eo[tp]': 'min',
                'eo[fp]': 'min',
                'eo': 'min'}


def calculate_equalized_odds_np(y_true, y_pred, groups):
    unique_groups = np.array([0, 1])
    tpr_groups = []
    fpr_groups = []

    for group in unique_groups:
        group_mask = groups == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        if group_y_true.size == 0:
            tpr_groups.append(0.0)
            fpr_groups.append(0.0)
            continue

        tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
        tpr_groups.append(tp / float(tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_groups.append(fp / float(fp + tn) if (fp + tn) > 0 else 0.0)

    return abs(tpr_groups[0] - tpr_groups[1]), abs(fpr_groups[0] - fpr_groups[1]), tpr_groups, fpr_groups


def auprc_score(y_true, y_score):
    """
    Calculate the area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def optimal_f1_score(y_true, y_score):
    """
    Calculate the optimal F1 score achievable by exhaustion.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall)
    optimal_f1_score = np.max(f1_scores)
    return optimal_f1_score


def get_optimal_f1_cutoff(y_scores, y_labels):
    """
    method to get the optimal cutoff for f1 score, also used in previous works.
    """
    cutoffs = np.linspace(0.01, 1.0, 1000)
    f1_scores = []
    for cutoff in cutoffs:
        y_preds = y_scores > cutoff
        f1 = f1_score(y_true=y_labels, y_pred=y_preds)
        f1_scores.append(f1)
    cutoff = cutoffs[np.argmax(f1_scores)]
    return cutoff


def get_optimal_err_cutoff(y_scores, y_labels):
    fpr, tpr, thresholds = roc_curve(y_labels, y_scores)
    optimal_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def get_selecting_metrics(metric_dict):
    return {
        "auroc": metric_dict["auroc"],
        "auprc": metric_dict["auprc"],
        "eo[tp]": metric_dict["eo[tp]"],
        "eo[fp]": metric_dict["eo[fp]"],
        "eo": metric_dict["eo"],
    }
