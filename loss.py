import torch
import torch.nn as nn
import torch.nn.functional as F


def energy_ranking(la, label, beta=0.1):
    # la : batch x length
    # label : batch x 1
    label = label.unsqueeze(-1)  # shape : batch x 1
    la = F.normalize(la)  # l2 norm equals to 1
    sim = (1.0 + la.mm(la.t())) / 2.  # similarity from 0 to 1

    target = 1 - (label == label.t()).float().reshape(1, -1)    # 0 : same category, 1: different category
    pair_potential = torch.exp(-beta * sim).reshape(1, -1)  # small: same category, large: different category

    # margin ranking
    # b x b negative: same category - different category, positive: different category - same category
    energy_diff = pair_potential - pair_potential.t()
    label_diff = torch.sign(target - target.t())  # b x b  0: no loss, -1: same-different, 1: different-same
    objective = -energy_diff * label_diff 
    loss_value = torch.sum((objective + torch.abs(objective)) / 2)  # sum of positive value
    loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))  # number of positive value
    loss = loss_value / (loss_num + 1e-10)
    return loss
