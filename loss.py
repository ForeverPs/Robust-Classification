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


def bit_energy_ranking(la, label, beta=1, log_space=False):
    # la : batch x length
    # label : batch x 1
    # c : batch x length x length

    # normalize
    label = label.unsqueeze(-1)  # shape : batch x 1
    la = F.normalize(la, p=1)  # l1 norm equals to 1
    dist = torch.sum(torch.abs(la[:,None,:] - la), dim=2)  # l1 distance

    target = 1 - (label == label.t()).float().reshape(1, -1)    # 0 : same category, 1: different category
    pair_potential = torch.exp(beta * dist).reshape(1, -1)  # small: same category, large: different category

    # margin ranking
    # b x b negative: same category - different category, positive: different category - same category
    energy_diff = pair_potential - pair_potential.t()
    label_diff = torch.sign(target - target.t())  # b x b  0: no loss, -1: same-different, 1: different-same
    objective = -energy_diff * label_diff 
    loss_value = torch.sum((objective + torch.abs(objective)) / 2)  # sum of positive value
    loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))  # number of positive value
    loss = loss_value / (loss_num + 1e-10)
    return loss


# def bit_energy_ranking(la, label, bits=2, beta=1., log_space=False):
#     # la : batch x length
#     # label : batch x 1
#     # c : batch x length x length

#     # normalize
#     b, k = la.shape
#     assert k % bits == 0
#     dim = k // bits
#     la1 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(0)  # 1 x batch x k
#     la2 = (la / torch.sum(la, dim=1).unsqueeze(-1)).unsqueeze(1)  # batch x 1 x k
#     la1 = la1.repeat((b, 1, 1)).reshape(b * b, k)
#     la2 = la2.repeat((1, b, 1)).reshape(b * b, k)
#     tensor_list = [torch.ones((bits, bits)) for _ in range(dim)]
#     c = (2 * torch.ones((k, k)) - torch.block_diag(*tensor_list)).repeat((b * b, 1, 1))
#     l = sinkhorn(c, la1, la2, log_space=log_space)

#     # 0 for same categories, 1 for different categories
#     target = 1- (label == label.t()).float().reshape(1, -1)
#     pair_potential = torch.exp(beta * l).reshape(1, -1)

#     # margin ranking
#     # energy > 0: diff - same, energy < 0: same - diff
#     energy_diff = pair_potential - pair_potential.t()  # b x b 
#     # -1: same - diff, 0: ignore, 1: diff - same
#     label_diff = torch.sign(target - target.t())  # b x b
#     objective = -energy_diff * label_diff
#     loss_value = torch.sum((objective + torch.abs(objective)) / 2)
#     loss_num = torch.sum(torch.sign(objective + torch.abs(objective)))
#     loss = loss_value / (loss_num + 1e-10)
#     return loss


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def sinkhorn_iterations(Z: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, iters: int) -> torch.Tensor:
    u, v = torch.ones_like(mu), torch.ones_like(nu)
    for _ in range(iters):
        u = mu / torch.einsum('bjk,bk->bj', [Z, v])
        v = nu / torch.einsum('bkj,bk->bj', [Z, u])
    return torch.einsum('bk,bkj,bj->bjk', [u, Z, v])


def sinkhorn(C, a, b, eps=2e-1, n_iter=10, log_space=True):
    """
    Args:
        a: tensor, normalized, note: no zero elements
        b: tensor, normalized, note: no zero elements
        C: cost Matrix [batch, n_dim, n_dim], note: no zero elements
    """
    P = torch.exp(-C/eps)
    if log_space:
        log_a = a.log()
        log_b = b.log()
        log_P = P.log()
    
        # solve the P
        log_P = log_sinkhorn_iterations(log_P, log_a, log_b, n_iter)
        P = torch.exp(log_P)
    else:
        P = sinkhorn_iterations(P, a, b, n_iter)
    return torch.sum(C * P, dim=[1, 2])
