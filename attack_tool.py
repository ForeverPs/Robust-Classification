import torch
import torch.nn as nn
from torch.autograd import Variable

def fgsm_attach(model, x, y, epsilon=0.05):
    model.eval()
    x.requires_grad_()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(x, y)
    loss.backward()
    x_grad = x.grad.data
    x.data += epsilon * torch.sign(x_grad)
    x_adv = x.detach()
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)
    return x_adv

def pgd_inf_attack(model, x, y, steps=10, step_size=0.002, epsilon=0.05):
    model.eval()
    kl_loss = nn.KLDivLoss(reduction='sum')
    x_adv = x.detach() + torch.randn_like(x.shape) * 0.001
    y_target = model(x.detach())
    y_target = torch.softmax(y_target, dim=1)
    x_adv.requires_grad_()
    for i in range(steps):
        y_pred = model(x_adv)
        y_pred = torch.log_softmax(y_pred, dim=1)
        loss = kl_loss(y_pred, y_target)
        loss.backward()
        x_adv.data += step_size * torch.sign(x_adv.grad.data)
        x_adv = torch.clip(x_adv, x - epsilon, x + epsilon)
        x_adv = torch.clip(x_adv, 0, 1)
        x_adv.grad.data.zero_()
    
    x_adv.requires_grad_(False).detach()
    return x_adv

# def pgd_l2_attack(model, x, y, steps=0, step_size=0.001, epsilon=0.05):
#     model.eval()
#     kl_loss = nn.KLDivLoss(reduction='sum')
#     x_adv = x.detach() + torch.randn_like(x.shape) * 0.001
#     y_target = model(x.detach())
#     y_target = torch.softmax(y_target, dim=1)


if __name__ == '__main__':
    # method 1
    x = torch.randn((2, 2))
    for i in range(10):
        x.requires_grad_()
        f = (x * x).sum()
        f.backward()
        x = x.data - 0.01 * x.grad.data
        # print(x)
        print(f)

    # method2
    print('*' * 20)
    x = torch.randn((2, 2))
    x = Variable(x, requires_grad=True)

    for i in range(10):
        f = (x * x).sum()
        f.backward()
        x.data = x.data - 0.01 * x.grad.data
        x.grad.data.zero_()
        print(f)
    
    # method3
    print('*' * 20)
    x = torch.randn((2, 2))
    x.requires_grad_()

    for i in range(10):
        f = (x * x).sum()
        f.backward()
        x.data = x.data - 0.01 * x.grad.data
        x.grad.data.zero_()
        print(f)
    
