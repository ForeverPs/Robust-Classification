import os
import tqdm
import torch
import shutil
import argparse
import random
import torch.nn as nn
from data_aug import *
from torch import optim, xlogy
from eval import get_acc
from utils import cutmix
from data import data_pipeline
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from model.se_resnet import se_resnet50, se_resnet18, se_resnet34
from attack_tool import fgsm_attack, pgd_inf_attack


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    epochs,
    batch_size,
    transform,
    lr=1e-3,
    train_image_txt='data/train_phase1/label.txt',
    val_image_txt='data/track1_test1/label.txt',
    eval_step = 2,
    cut_mix=0.5,
    checkpoint_path=''
):
    train_loader, val_loader = data_pipeline(
        train_image_txt, val_image_txt, transform, batch_size)

    model = se_resnet18(num_classes=20)
    model = nn.DataParallel(model).to(device)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=.9, weight_decay=1e-4)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 20 * len(train_loader.dataset) // batch_size, 1e-7)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.8
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
                loss = criterion(predict, target_a) * lam + \
                    criterion(predict, target_b) * (1. - lam)
            
            # adv train
            fgsm_x = fgsm_attack(model, x, y)
            predict = model(fgsm_x)
            fgsm_loss = criterion(predict, y)

            # pgd_x = pgd_inf_attack(model, x, y)
            # predict = model(pgd_x)
            # pgd_loss = criterion(predict, y)

            loss = loss + 0.1 * fgsm_loss # + 0.1 * pgd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            train_acc += get_acc(predict_cls, y)

        # update learning rate
        scheduler.step()

        with torch.no_grad():
            if epoch % eval_step != 0:
                continue
            model.eval()
            for x, y in tqdm.tqdm(val_loader):
                x = x.float().to(device)
                y = y.long().to(device)
                predict = model(x)
                loss = criterion(predict, y)
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
            torch.save(model.state_dict(), checkpoint_path + '/%s.pth' %
                       model_name)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    # logdir = './tensorboard/SeResNet18/'
    # shutil.rmtree(logdir, True)
    # writer = SummaryWriter(log_dir=logdir)

    parser = argparse.ArgumentParser()
    # 指定保存特征字
    parser.add_argument("--name", type=str, default="", help='log_name')
    opt = parser.parse_args()

    # 根据特征字和当前时间确定目录名字
    import time
    cur_t = time.strftime('%Y-%m-%d', time.localtime())
    log_path = f'./log/log_{opt.name}_{cur_t}'
    checkpoint_path = f'./log/checkpoint_{opt.name}_{cur_t}'


    # 创建目录名字，并且备份当前代码
    writer = SummaryWriter(log_dir=log_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.system(f'cp ./*.py {log_path}')
    os.system(f'cp -r model {log_path}')

    lr = 1e-3
    epochs = 3000
    batch_size = 64
    train_image_txt = 'data/train_phase1/label.txt'
    val_image_txt = 'data/track1_test1/label.txt'

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
    ])

    train(epochs, batch_size, transform, lr=lr,
          train_image_txt=train_image_txt, val_image_txt=val_image_txt, cut_mix=0.3, checkpoint_path=checkpoint_path)
