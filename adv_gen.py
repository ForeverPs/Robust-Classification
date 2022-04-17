import os
import torch
# import shutil
import numpy as np
import torch.nn as nn
# from data import data_pipeline
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# from model.se_resnet import SmoothBitSeResNetML
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = SmoothBitSeResNetML(depth=18, num_classes=20, num_bit=6)
# model_path = '/opt/tiger/debug_server/Robust-Classification/saved_models/smooth_bit_bw/epoch_393_acc_1.000.pth'

# remove module prefix
# new_dict = dict()
# for k, v in torch.load(model_path, map_location='cpu').items():
#     new_dict[k.replace('module.', '')] = v
# model.load_state_dict(new_dict, strict=True)

# model = model.to(device).eval()
# print('Successfully load trained model...')

# # freeze model
# for param in model.parameters():
#     param.requires_grad = False


def fgsm_attack(model, x, y, T=1, epsilon=None, start=0.001):
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
    x_adv = torch.clip(x_adv, 0, 1)

    x_adv2 = x.data - epsilon * x.grad.data.sign()  # noise
    x_adv2 = torch.clip(x_adv2, 0, 1)
    x_adv2.requires_grad_(False)

    x_adv = torch.cat([x_adv, x_adv2], dim=0)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


def target_fgsm_attack(model, x, T=1, num_classes=20, epsilon=None, start=0.001):
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
    x_adv = torch.clip(x_adv, 0, 1)

    x_adv2 = x.data + epsilon * x.grad.data.sign()  # noise
    x_adv2 = torch.clip(x_adv2, 0, 1)
    x_adv2.requires_grad_(False)

    x_adv = torch.cat([x_adv, x_adv2], dim=0)

    # open the model for training
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    return x_adv


# def freeze_fgsm_attack(x, y, T=1, start=0.001):
#     # freeze parameters for fast forward
#     x = x.detach()
#     epsilon = np.random.choice(np.linspace(start, 0.08, num=20), size=1)[0]
#     x.requires_grad_(True)
#     pre = model(x)
#     loss = nn.CrossEntropyLoss()(pre / T, y)
#     model.zero_grad()  # empty grad
#     loss.backward()

#     x_adv = x.data + epsilon * x.grad.data.sign()  # gradient ascend
#     x_adv = torch.clip(x_adv, 0, 1)
#     x_adv.requires_grad_(False)
#     return x_adv


# def freeze_target_fgsm_attack(x, T=1, num_classes=20, start=0.001):
#     x = x.detach()  # batch, 3, w, h
#     y = torch.randint(0, num_classes, size=(x.shape[0],)).to(x.device)
#     epsilon = np.random.choice(np.linspace(start, 0.08, num=20), size=1)[0]
#     x.requires_grad_(True)
#     pre = model(x)
#     loss = nn.CrossEntropyLoss()(pre / T, y)
#     model.zero_grad()  # empty grad
#     loss.backward()

#     x_adv = x.data - epsilon * x.grad.data.sign()  # gradient descent
#     x_adv = torch.clip(x_adv, 0, 1)
#     x_adv.requires_grad_(False)
#     return x_adv


# if __name__ == '__main__':
#     image_txt='data/train_phase1/label.txt'
#     batch_size = 1
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])

#     all_loader, train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)

#     count = 0
#     shutil.rmtree('adv_samples/', True)
#     os.makedirs('adv_samples', exist_ok=True)
#     for x, y in train_loader:
#         original = x.squeeze(0).permute(1, 2, 0)
#         x_adv = freeze_target_fgsm_attack(x.to(device), y.to(device)).squeeze(0).permute(1, 2, 0).cpu().numpy()
#         new_img = np.concatenate([original, x_adv], axis=1)
#         print(new_img.shape)
#         save_name = 'adv_samples/%d.jpg' % count
#         count += 1
#         plt.imsave(save_name, new_img)
