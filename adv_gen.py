import os
import torch
import shutil
import numpy as np
import torch.nn as nn
from data import data_pipeline
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.efficient_net import EfficientNet

def fgsm_attack(model, x, y):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    _, pre = model(x)
    loss = nn.CrossEntropyLoss()(pre, y)
    model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data + epsilon * x.grad.data.sign()
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


def target_fgsm_attack(model, x):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()  # batch, 3, w, h
    y = torch.randint(0, 20, size=(x.shape[0],)).to(x.device)
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    _, pre = model(x)
    loss = nn.CrossEntropyLoss()(pre, y)
    model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data - epsilon * x.grad.data.sign()  # gradient descent
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


if __name__ == '__main__':
    model = EfficientNet.from_name(model_name='efficientnet-b4', num_classes=20, dropout_rate=0.3)
    model_path = '/opt/tiger/debug_server/Robust-Classification/saved_models/efficient/epoch_9_acc_0.821.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    image_txt='data/train_phase1/label.txt'
    batch_size = 1
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)

    count = 0
    shutil.rmtree('adv_samples/', True)
    os.makedirs('adv_samples', exist_ok=True)
    for x, y in train_loader:
        # x_adv = fgsm_attack(model, x, y).squeeze(0).permute(1, 2, 0).numpy()
        x_adv = target_fgsm_attack(model, x).squeeze(0).permute(1, 2, 0).numpy()
        save_name = 'adv_samples/%d.jpg' % count
        count += 1
        plt.imsave(save_name, x_adv)
        print(x_adv.shape)
