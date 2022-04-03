import os
import tqdm
import torch
import shutil
import random
import torch.nn as nn
from data_aug import *
from torch import optim
from eval import get_acc
from data import data_pipeline
from loss import energy_ranking
from model.se_resnet import SeResNet
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from adv_gen import fgsm_attack, target_fgsm_attack
from utils import cutmix, mixup_data, mixup_criterion


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, transform, lr=1e-3, image_txt='data/train_phase1/label.txt', cut_mix=0.0):
    all_loader, train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)

    model = SeResNet(depth=18, num_classes=20, dropout=0.1)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        print('Training with a Pretrained Model...')
    except:
        print('Training from Scratch...')
    
    model = model.to(device)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=.9, weight_decay=1e-4)
    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.97
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        model.train()
        for x, y in tqdm.tqdm(all_loader):
            x = x.float().to(device)
            y = y.long().to(device)

            # Cut mix training
            if random.uniform(0, 1) >= cut_mix:
                feat, predict = model(x)
                loss_ = criterion(predict, y) + 0.1 * energy_ranking(feat, y)
            else:
                x, target_a, target_b, lam = cutmix(x, y)
                feat, predict = model(x)
                loss_ = criterion(predict, target_a) * lam + criterion(predict, target_b) * (1. - lam)
            
            # predict result on clean images
            _, predict_cls = torch.max(predict, dim=-1)
            
            # Mix up training
            # inputs, targets_a, targets_b, lam = mixup_data(x, y, 1, use_cuda=True)
            # inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            # _, outputs = model(inputs)
            # loss_mixup = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Untarget FGSM training
            x_adv = fgsm_attack(model, x.clone(), y, T=1)
            _, predict_adv = model(x_adv.to(device))
            # loss_adv = criterion(predict_adv, y)
            loss_adv = criterion(predict_adv, predict_cls)

            # Target FGSM training
            x_adv = target_fgsm_attack(model, x.clone(), T=1)
            _, predict_adv_target = model(x_adv.to(device))
            # loss_adv_target = criterion(predict_adv_target, y)
            loss_adv_target = criterion(predict_adv_target, predict_cls)

            loss = loss_ + adv_weight * loss_adv + adv_weight * loss_adv_target # + 0.3 * loss_mixup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

            # _, predict_cls = torch.max(predict, dim=-1)
            _, predict_cls_adv = torch.max(predict_adv, dim=-1)
            _, predict_cls_adv_target = torch.max(predict_adv_target, dim=-1)
            train_acc += (get_acc(predict_cls, y) + get_acc(predict_cls_adv, y) * adv_weight + get_acc(predict_cls_adv_target, y) * adv_weight) / (1 + 2 * adv_weight)

        # update learning rate
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader):
                x = x.float().to(device)
                y = y.long().to(device)
                feat, predict = model(x)
                loss = criterion(predict, y)

                val_loss = val_loss + loss.item()

                _, predict_cls = torch.max(predict, dim=-1)
                val_acc += get_acc(predict_cls, y)

        train_loss = train_loss / len(all_loader)
        train_acc = train_acc / len(all_loader)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | Val Acc : %.3f'
              % (epoch, train_loss, train_acc, val_loss, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            model_name = 'epoch_%d_acc_%.3f' % (epoch, val_acc)
            os.makedirs('./saved_models/energy_ranking_seres18/', exist_ok=True)
            torch.save(model.state_dict(), './saved_models/energy_ranking_seres18/%s.pth' % model_name)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    # model_path = 'saved_models/energy_ranking_seres18/epoch_2_acc_0.978.pth' 
    model_path = None
    logdir = './tensorboard/SeResNet18/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir=logdir)

    lr = 1e-3
    epochs = 3000
    batch_size = 64
    adv_weight = 1  # TRADES: 1 or 6
    image_txt = 'data/train_phase1/label.txt'

    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224), 
        PepperSaltNoise(p=0.15),
        ColorPointNoise(p=0.15),
        GaussianNoise(p=0.15),
        Mosaic(p=0.2),
        RGBShuffle(p=0.05),
        Rotate(p=0.1),
        HFlip(p=0.1),
        VFlip(p=0.05),
        MotionBlur(p=0.1),
        GaussianBlur(p=0.01),
        Blur(p=0.01),
        Rain(p=0.1),
        Extend(p=0.05),
        BlockShuffle(p=0.1),
        LocalShuffle(p=0.05),
        RandomPadding(p=0.2),
        transforms.ToTensor()
    ])

    train(epochs, batch_size, transform, lr=lr, image_txt=image_txt, cut_mix=0.0)
