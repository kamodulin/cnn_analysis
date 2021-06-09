import torch


def accuracy_score(y_true, y_pred):
    accuracy_bool = torch.eq(y_true.unsqueeze(dim=1), y_pred).any(dim=1)
    return accuracy_bool.int().mean()


# def weight_similarity():