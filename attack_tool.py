import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


    


def fgsm_attack(model, x, y, epsilon=0.00):

    for p in model.parameters():
        p.requires_grad = False
    x = x.detach()

    x.requires_grad_()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(model(x), y)
    loss.backward()
    # print(x.requires_grad)
    x_grad = x.grad.data
    x.data += epsilon * torch.sign(x_grad.data)
    x_adv = x.detach()
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)

    for p in model.parameters():
        p.requires_grad = True
    return x_adv

def pgd_inf_attack(model, x, y, steps=10, step_size=0.002, epsilon=0.05):
    model.eval()
    x = x.detach()

    x.requires_grad = False
    kl_loss = nn.KLDivLoss(reduction='sum')
    x_adv = x.detach().data + torch.randn_like(x) * 0.001
    y_target = model(x.detach())
    y_target = torch.softmax(y_target, dim=1).detach()
   
    for i in range(steps):
        x_adv.requires_grad_()

        y_pred = torch.log_softmax(model(x_adv), dim=1)
        loss = kl_loss(y_pred, y_target)
        loss.backward()

        x_adv.data += step_size * torch.sign(x_adv.grad.data)
        x_adv.grad.data.zero_()

        with torch.no_grad():
            x_adv = torch.clip(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clip(x_adv, 0, 1)
           
    x_adv.requires_grad_(False).detach()
    model.train()
    return x_adv




def pgd_window_attack(model, x, y, window_size=(32, 32), n_iter=10, step_size=0.03):
    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    cols = torch.arange(window_size[0]) + torch.randint(h-window_size[0], (1,))
    rows = torch.arange(window_size[1]) + torch.randint(w-window_size[1], (1,))
    
    cols, rows = torch.meshgrid(cols, rows, indexing='ij')
    
    cols = cols.reshape(-1)
    rows = rows.reshape(-1)
    
    noise = torch.randn((b, c, window_size[0] * window_size[1])) * step_size

    noise.requires_grad_()
    for i in range(n_iter):
        # add up the noise
        x = x.detach()
        
        x[:, :, cols, rows] = noise

        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(model(x), y)

        loss.backward()

        noise_grad = noise.grad.data
    
        noise.data = noise.data + step_size * torch.sign(noise_grad)
    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()



def pgd_pixels_attack(model, x, y, nums, n_iter=10, step_size=0.07):

    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    cols = torch.randint(h, (nums,))
    rows = torch.randint(w, (nums,))
    
    noise = torch.randn((b, c, nums)) * step_size

    noise.requires_grad_()
    
    for i in range(n_iter):
        x = x.detach()
        # add up the noise
        x[:, :, cols, rows] = noise

        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(model(x), y)

        loss.backward()

        noise_grad = noise.grad.data
    
        noise.data = noise.data + step_size * torch.sign(noise_grad)
    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()




def pgd_lines_attack(model, x, y, nums, direction='vertical', n_iter=10, step_size=0.07):
    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    if direction == 'vertical':
        cols = torch.randperm(h)[:nums]
        rows = torch.arange(w)
        noise = torch.randn((b, c, nums * w)) * step_size
    else:
        cols = torch.arange(h)
        rows = torch.randperm(w)[:nums]
        noise = torch.randn((b, c, nums * h)) * step_size
    
    cols, rows = torch.meshgrid(cols, rows, indexing='ij')
    
    cols = cols.reshape(-1)
    rows = rows.reshape(-1)
    noise.requires_grad_()
    
    for i in range(n_iter):
        x = x.detach()
        
        # add up the noise
        x[:, :, cols, rows] = noise

        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(model(x), y)

        loss.backward()

        noise_grad = noise.grad.data

        noise.data = noise.data + step_size * torch.sign(noise_grad)
        
        noise.grad.data.zero_()

    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()



def fgsm_lines_attack(model, x, y, nums, epsilon=0.5, direction='vertical'):
    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    if direction == 'vertical':
        cols = torch.randperm(h)[:nums]
        rows = torch.arange(w)
        noise = torch.randn((b, c, nums * w)) * epsilon
    else:
        cols = torch.arange(h)
        rows = torch.randperm(w)[:nums]
        noise = torch.randn((b, c, nums * h)) * epsilon
    
    cols, rows = torch.meshgrid(cols, rows, indexing='ij')
    
    cols = cols.reshape(-1)
    rows = rows.reshape(-1)
    
    

    noise.requires_grad_()
    
    # add up the noise
    x[:, :, cols, rows] = noise

    loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(model(x), y)
    
    loss.backward()
    
    noise_grad = noise.grad.data

    noise.data += epsilon * torch.sign(noise_grad)
    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()



def fgsm_pixels_attack(model, x, y, nums=20, epsilon=0.5):
    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    cols = torch.randint(h, (nums,))
    rows = torch.randint(w, (nums,))
    
    noise = torch.randn((b, c, nums)) * epsilon

    noise.requires_grad_()
    
    # add up the noise
    x[:, :, cols, rows] = noise

    loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(model(x), y)
    
    loss.backward()
    
    noise_grad = noise.grad.data
#     x.data[:, :, idxs, :] += epsilon * torch.sign(x_grad.data)[:, :, idxs, :]
    noise.data += epsilon * torch.sign(noise_grad)
    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()



def fgsm_window_attack(model, x, y, window_size=(32, 32), epsilon=0.3):
    x = x.clone()
    # set the model grad false
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = False
    
    b, c, h, w = x.shape
    
    cols = torch.arange(window_size[0]) + torch.randint(h-window_size[0], (1,))
    rows = torch.arange(window_size[1]) + torch.randint(w-window_size[1], (1,))
    
    cols, rows = torch.meshgrid(cols, rows, indexing='ij')
    
    cols = cols.reshape(-1)
    rows = rows.reshape(-1)
    
    noise = torch.randn((b, c, window_size[0] * window_size[1])) * epsilon

    noise.requires_grad_()
    
    # add up the noise
    x[:, :, cols, rows] = noise

    loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(model(x), y)
    
    loss.backward()
    
    noise_grad = noise.grad.data

    noise.data += epsilon * torch.sign(noise_grad)
    
    x[:, :, cols, rows] = noise
    
    x = torch.clip(x, 0, 1)

    for p in model.parameters():
        p.requires_grad = True
        
    return x.detach()


if __name__ == '__main__':
    # method 1

    x = torch.randn((4, 3, 224, 224))
    y = torch.randint(0, 20, (4,))
    from model.se_resnet import se_resnet18
    model = se_resnet18()
    x_adv = fgsm_attack(model, x, y)

    x_adv = pgd_inf_attack(model, x, y)

    # x = torch.randn((2, 2))
    # for i in range(10):
    #     x.requires_grad_()
    #     f = (x * x).sum()
    #     f.backward()
    #     x = x.data - 0.01 * x.grad.data
    #     # print(x)
    #     print(f)

    # # method2
    # print('*' * 20)
    # x = torch.randn((2, 2))
    # x = Variable(x, requires_grad=True)

    # for i in range(10):
    #     f = (x * x).sum()
    #     f.backward()
    #     x.data = x.data - 0.01 * x.grad.data
    #     x.grad.data.zero_()
    #     print(f)
    
    # # method3
    # print('*' * 20)
    # x = torch.randn((2, 2))
    # x.requires_grad_()

    # for i in range(10):
    #     f = (x * x).sum()
    #     f.backward()
    #     x.data = x.data - 0.01 * x.grad.data
    #     x.grad.data.zero_()
    #     print(f)
    
