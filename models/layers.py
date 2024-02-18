import torch

def kl_divergence(predict, actual, epsilon=10**-15):
    predict = torch.clip(predict, min=epsilon, max=1-epsilon)
    l = (actual * torch.log(actual/(predict))).reshape(-1)
    mask = torch.isnan(l)
    return l[~mask].mean()