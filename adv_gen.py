import torch
import numpy as np
import torch.nn as nn


def fgsm_attack(model, x, y, T=1, epsilon=None, start=0.001, gradient=False):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()
    if epsilon is None:
        epsilon = np.random.choice(np.linspace(start, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data + epsilon * x.grad.data.sign()  # gradient ascend
    x_adv = torch.clamp(x_adv, 0, 1)
    if not gradient:
        return x_adv
    else:
        return x_adv, x.grad.data.sign()


def target_fgsm_attack(model, x, T=1, num_classes=20, epsilon=None, start=0.001, gradient=False):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()  # batch, 3, w, h
    y = torch.randint(0, num_classes, size=(x.shape[0],)).to(x.device)
    if epsilon is None:
        epsilon = np.random.choice(np.linspace(start, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data - epsilon * x.grad.data.sign()  # gradient descent
    x_adv = torch.clamp(x_adv, 0, 1)

    if not gradient:
        return x_adv
    else:
        return x_adv, x.grad.data.sign()
