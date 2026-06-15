import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from utils.metrics import calculate_equalized_odds_np, auprc_score, get_optimal_err_cutoff
from utils.mapper import label2onehot


@torch.no_grad()
def test_fn(model, train_config, val_loader, test_loader):
    print(f"Start Testing on Seed {train_config.seed}")
    model.to(train_config.device)
    model.eval()

    num_classes = train_config.num_classes
    num_sens = train_config.num_sens

    # get cutoff from val set
    if train_config.cutoff is None:
        val_metric_dict = _test_loop(model, loader=val_loader, device=train_config.device,
                                    num_classes=num_classes, num_sens=num_sens, cutoff=None)
        cutoff = val_metric_dict["cutoff"]

    else:
        cutoff = train_config.cutoff
        
    # test
    metric_dict = _test_loop(model, loader=test_loader, device=train_config.device,
                             num_classes=num_classes, num_sens=num_sens, cutoff=cutoff)
                
    return metric_dict


@torch.no_grad()
def _test_loop(model, loader, device,
            num_classes: int, num_sens: int,
            cutoff=None):
    print("Testing Loop")
    model.eval()

    # collections
    y_scores_collection = []
    labels_collection = []
    sensitives_collection = []

    for images, labels, sensitives in loader:
        labels = label2onehot(labels, num_classes=num_classes)
        sensitives = label2onehot(sensitives, num_classes=num_sens)
        images, labels = images.to(device), labels.to(device)
        y_logits, _ = model.forward_with_hidden(images)

        # eval
        y_logits = torch.softmax(y_logits, dim=1).detach().cpu().numpy()
        batch_y_score = y_logits[:, 1]  # take the prob of positive class as score
        batch_labels = torch.argmax(labels, dim=1).cpu().numpy().astype(np.int8)
        batch_sensitives = torch.argmax(sensitives, dim=1).cpu().numpy().astype(np.int8)

        y_scores_collection.append(batch_y_score)
        labels_collection.append(batch_labels)
        sensitives_collection.append(batch_sensitives)
    
    y_scores = np.concatenate(y_scores_collection)
    labels = np.concatenate(labels_collection)
    sensitives = np.concatenate(sensitives_collection)

    auprc = auprc_score(y_true=labels, y_score=y_scores)
    auroc = roc_auc_score(y_true=labels, y_score=y_scores)

    # get y_preds from y_scores
    if cutoff is None:
        cutoff = get_optimal_err_cutoff(y_scores=y_scores, y_labels=labels)
    y_preds = y_scores > cutoff

    f1 = f1_score(y_true=labels, y_pred=y_preds)
    tpr_discrepancy, fpr_discrepancy, tpr_groups, fpr_groups = calculate_equalized_odds_np(y_true=labels,
                                                                                           y_pred=y_preds,
                                                                                           groups=sensitives)

    eo = max(tpr_discrepancy, fpr_discrepancy)
    metric_dict = {"auroc": auroc,
                   "auprc": auprc,
                   "f1": f1,
                   "eo[tp]": tpr_discrepancy,
                   "eo[fp]": fpr_discrepancy,
                   "eo": eo,
                   "cutoff": cutoff}
    for i_group in range(len(tpr_groups)):
        if i_group == 0:
            metric_dict["tpr_male"] = tpr_groups[i_group]
            metric_dict["fpr_male"] = fpr_groups[i_group]
        elif i_group == 1:
            metric_dict["tpr_female"] = tpr_groups[i_group]
            metric_dict["fpr_female"] = fpr_groups[i_group]
        else:
            raise ValueError("Only support 2 gender groups")

    return metric_dict
