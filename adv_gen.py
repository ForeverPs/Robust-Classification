import os
import torch
import shutil
import numpy as np
import torch.nn as nn
from data import data_pipeline
import matplotlib.pyplot as plt
from model.bit_resnet import resnet18, resnet50
import torchvision.transforms as transforms
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model = resnet18(num_classes=20)
model_path = 'log/checkpoint_bit_resnet_gelu_in_attack_2022-04-04/epoch_298_acc_0.854.pth'

# remove module prefix
new_dict = dict()
for k, v in torch.load(model_path, map_location='cpu').items():
    new_dict[k.replace('module.', '')] = v
model.load_state_dict(new_dict, strict=True)

model = model.to(device).eval()
print('Successfully load trained model...')

# freeze model
for param in model.parameters():
    param.requires_grad = False


def fgsm_attack(model, x, y, T=1):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    _, pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data + epsilon * x.grad.data.sign()  # gradient ascend
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


def target_fgsm_attack(model, x, T=1, num_classes=20):
    # freeze parameters for fast forward
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = x.detach()  # batch, 3, w, h
    y = torch.randint(0, num_classes, size=(x.shape[0],)).to(x.device)
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    _, pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    # model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data - epsilon * x.grad.data.sign()  # gradient descent
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


def freeze_fgsm_attack(x, y, T=1):
    # freeze parameters for fast forward
    x = x.detach()
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    # model.zero_grad()  # empty grad
    loss.backward()

    x_adv = x.data + epsilon * x.grad.data.sign()  # gradient ascend
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)
    return x_adv


def freeze_target_fgsm_attack(x, T=1, num_classes=20):
    x = x.detach()  # batch, 3, w, h
    y = torch.randint(0, num_classes, size=(x.shape[0],)).to(x.device)
    epsilon = np.random.choice(np.linspace(0.01, 0.08, num=20), size=1)[0]
    x.requires_grad_(True)
    pre = model(x)
    loss = nn.CrossEntropyLoss()(pre / T, y)
    model.zero_grad()  # empty grad
    # print(loss.requires_grad)
    loss.backward()

    x_adv = x.data - epsilon * x.grad.data.sign()  # gradient descent
    x_adv = torch.clip(x_adv, 0, 1)
    x_adv.requires_grad_(False)
    return x_adv


if __name__ == '__main__':
    # load model
    # model = SeResNet(depth=18, num_classes=20, dropout=0.1)
    # model_path = '/opt/tiger/debug_server/Robust-Classification/saved_models/energy_ranking_seres18/epoch_263_acc_0.989.pth'
    # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    # model = model.eval()

    # load model
    # model_path = 'saved_models/bit_energy_all/epoch_128_acc_0.974.pth'
    # model = BitSeResNetML(depth=18, num_classes=20, num_bit=6)

    # # remove module prefix
    # new_dict = dict()
    # for k, v in torch.load(model_path, map_location='cpu').items():
    #     new_dict[k.replace('module.', '')] = v
    # model.load_state_dict(new_dict, strict=True)
    # model = model.eval()
    # print('Successfully load trained model...')


    image_txt='data/train_phase1/label.txt'
    batch_size = 1
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_loader = data_pipeline(image_txt, '', transform, batch_size, True)

    count = 0
    shutil.rmtree('adv_samples/', True)
    os.makedirs('adv_samples', exist_ok=True)
    for x, y in tqdm.tqdm(train_loader):
        original = x.squeeze(0).permute(1, 2, 0)
        # x_adv = freeze_fgsm_attack(x.to(device), y.to(device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        x_adv = freeze_target_fgsm_attack(x.to(device), y.to(device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        new_img = x_adv
        # print(new_img.shape)

        # x_adv = target_fgsm_attack(model, x).squeeze(0).permute(1, 2, 0).numpy()
        save_name = 'adv_samples/%d.png' % count
        count += 1
        plt.imsave(save_name, new_img)
        # plt.imsave(save_name, x_adv)
        # print(x_adv.shape)
