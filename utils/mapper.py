import torch


def map_sex(gender_str):
    if gender_str == "male":
        return torch.tensor(0)
    if gender_str == "female":
        return torch.tensor(1)
    raise ValueError(f"Unknown sex value {gender_str}")


def label2onehot(label, num_classes):
    label = label.unsqueeze(1).to(torch.int64)
    onehot = torch.zeros(label.size(0), num_classes).to(label.device)
    onehot.scatter_(1, label, 1)
    return onehot
