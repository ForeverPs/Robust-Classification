import os
import tqdm
import torch
import shutil
import random
import torch.nn as nn
from data_aug import *
from torch import optim
from eval import get_acc
from utils import cutmix
from data import data_pipeline
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from model.se_resnet import se_resnet50, se_resnet18, se_resnet34


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, transform, lr=1e-3, image_txt='data/train_phase1/label.txt', cut_mix=0.5):
    train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)

    # model = se_resnet50(num_classes=20)
    # model = se_resnet34(num_classes=20)
    model = se_resnet18(num_classes=20)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.98
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        model.train()
        for x, y in tqdm.tqdm(train_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            if random.uniform(0, 1) > cut_mix:
                predict = model(x)
                loss = criterion(predict, y)
            else:
                x, target_a, target_b, lam = cutmix(x, y)
                predict = model(x)
                loss = criterion(predict, target_a) * lam + criterion(predict, target_b) * (1. - lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            train_acc += get_acc(predict_cls, y)

        # update learning rate
        scheduler.step()

        model.eval()
        for x, y in tqdm.tqdm(val_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            predict = model(x)
            loss = criterion(predict, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_loss = val_loss + loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            val_acc += get_acc(predict_cls, y)

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | Val Acc : %.3f'
              % (epoch, train_loss, train_acc, val_loss, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            model_name = 'epoch_%d_acc_%.3f' % (epoch, val_acc)
            torch.save(model.state_dict(), './saved_models/%s.pth' % model_name)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    logdir = './tensorboard/SeResNet18/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir=logdir)

    lr = 1e-3
    epochs = 3000
    batch_size = 64
    image_txt = 'data/train_phase1/label.txt'

    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        PepperSaltNoise(p=0.15),
        ColorPointNoise(p=0.15),
        GaussianNoise(p=0.15),
        Mosaic(p=0.15),
        RGBShuffle(p=0.05),
        Rotate(p=0.1),
        HFlip(p=0.1),
        VFlip(p=0.05),
        RandomCut(p=0.1),
        MotionBlur(p=0.1),
        GaussianBlur(p=0.01),
        Blur(p=0.01),
        Rain(p=0.15),
        # transforms.Resize(224),
        transforms.RandomResizedCrop(224), 
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train(epochs, batch_size, transform, lr=lr, image_txt=image_txt, cut_mix=0.3)
