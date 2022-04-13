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
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from model.se_resnet import SmoothBitSeResNetML
from utils import cutmix, mixup_data, mixup_criterion


os.environ['CUDA_VISIBLE_DEVICES'] = '0,4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from adv_gen import freeze_fgsm_attack, freeze_target_fgsm_attack, fgsm_attack, target_fgsm_attack


def train(epochs, batch_size, transform, lr=1e-3, image_txt='data/train_phase1/label.txt', cut_mix=0.0):
    all_loader, train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)

    model = SmoothBitSeResNetML(depth=18, num_classes=20, num_bit=6)
    try:
        # remove module prefix
        new_dict = dict()
        for k, v in torch.load(model_path, map_location='cpu').items():
            new_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_dict, strict=True)
        print('Training with a Trained Model...')
    except:
        print('Training from Scratch...')

    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.95
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc, target_acc, untarget_acc = 0, 0, 0, 0

        model.train()
        # for x, y in tqdm.tqdm(train_loader):
        for x, y in tqdm.tqdm(all_loader):
            x = x.float().to(device)
            y = y.long().to(device)

            # Cut mix training
            if random.uniform(0, 1) >= cut_mix:
                predict = model(x)
                loss_ = criterion(predict, y)
            else:
                x, target_a, target_b, lam = cutmix(x, y)
                predict = model(x)
                loss_ = criterion(predict, target_a) * lam + criterion(predict, target_b) * (1. - lam)
            
            # Mix Up training
            inputs, targets_a, targets_b, lam = mixup_data(x, y, 1, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss_mixup = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            double_y = torch.cat([y, y], dim=0)

            # Adversarial Training
            # White-Box
            # Untarget FGSM training
            x_adv = fgsm_attack(model, x.clone(), y)
            predict_adv_white = model(x_adv.to(device))
            loss_adv_white = criterion(predict_adv_white, double_y)

            # Target FGSM training
            x_adv = target_fgsm_attack(model, x.clone())
            predict_adv_target_white = model(x_adv.to(device))
            loss_adv_target_white = criterion(predict_adv_target_white, double_y)
            
            # Black-Box
            # Untarget FGSM training
            x_adv = freeze_fgsm_attack(x.clone(), y)
            predict_adv = model(x_adv.to(device))
            loss_adv = criterion(predict_adv, y)

            # Target FGSM training
            x_adv = freeze_target_fgsm_attack(x.clone())
            predict_adv_target = model(x_adv.to(device))
            loss_adv_target = criterion(predict_adv_target, y)

            # TRADES
            loss = loss_ + loss_mixup + adv_weight * loss_adv + adv_weight * loss_adv_target + adv_weight * loss_adv_white + adv_weight * loss_adv_target_white

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            _, predict_cls_adv = torch.max(predict_adv, dim=-1)
            _, predict_cls_adv_target = torch.max(predict_adv_target, dim=-1)
            _, predict_cls_adv_white = torch.max(predict_adv_white, dim=-1)
            _, predict_cls_adv_target_white = torch.max(predict_adv_target_white, dim=-1)
            train_acc += ((get_acc(predict_cls, y) + get_acc(predict_cls_adv, y) * adv_weight + get_acc(predict_cls_adv_target, y) * adv_weight + get_acc(predict_cls_adv_target_white, double_y) * adv_weight + get_acc(predict_cls_adv_white, double_y) * adv_weight) / (1 + 4 * adv_weight))

        # update learning rate
        scheduler.step()

        model.eval()
        for x, y in tqdm.tqdm(val_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            double_y = torch.cat([y, y], dim=0)

            # White-Box
            # Untarget FGSM training
            x_adv = fgsm_attack(model, x.clone(), y, start=0.03)
            predict_adv_white = model(x_adv.to(device))
            loss_adv_white = criterion(predict_adv_white, double_y)

            # Target FGSM training
            x_adv = target_fgsm_attack(model, x.clone(), start=0.03)
            predict_adv_target_white = model(x_adv.to(device))
            loss_adv_target_white = criterion(predict_adv_target_white, double_y)

            # original image
            predict = model(x)
            loss = criterion(predict, y)

            val_loss = (val_loss + loss.item() + loss_adv_white.item() + loss_adv_target_white.item()) / 3.0

            _, predict_cls = torch.max(predict, dim=-1)
            _, predict_cls_adv_white = torch.max(predict_adv_white, dim=-1)
            _, predict_cls_adv_target_white = torch.max(predict_adv_target_white, dim=-1)
            untarget_acc += get_acc(predict_cls_adv_white, double_y)
            target_acc += get_acc(predict_cls_adv_target_white, double_y)
            val_acc += get_acc(predict_cls, y)

        train_loss = train_loss / len(all_loader)
        train_acc = train_acc / len(all_loader)

        # train_loss = train_loss / len(train_loader)
        # train_acc = train_acc / len(train_loader)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        target_acc = target_acc / len(val_loader)
        untarget_acc = untarget_acc / len(val_loader)

        print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | Val Acc : %.3f | Val Target Attack Acc : %.3f | Val Untarget Attack Acc : %.3f '
              % (epoch, train_loss, train_acc, val_loss, val_acc, target_acc, untarget_acc))
            
        total_val_acc = (val_acc + target_acc + untarget_acc) / 3.0

        if total_val_acc >= best_acc:
            best_acc = total_val_acc
            model_name = 'epoch_%d_acc_%.3f' % (epoch, total_val_acc)
            os.makedirs('./saved_models/smooth_bit_bw/', exist_ok=True)
            torch.save(model.module.state_dict(), './saved_models/smooth_bit_bw/%s.pth' % model_name)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/target_acc', target_acc, epoch)
        writer.add_scalar('val/untarget_acc', untarget_acc, epoch)


if __name__ == '__main__':
    model_path = '/opt/tiger/debug_server/Robust-Classification/saved_models/smooth_bit_bw/epoch_393_acc_1.000.pth'
    logdir = './tensorboard/MedianSmoothBit/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir=logdir)

    lr = 1e-3
    epochs = 3000
    batch_size = 128
    adv_weight = 1  # TRADES : 1 or 6
    image_txt = 'data/train_phase1/label.txt'

    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224), 
        PepperSaltNoise(p=0.15),
        ColorPointNoise(p=0.15),
        GaussianNoise(p=0.15),
        Mosaic(p=0.15),
        RGBShuffle(p=0.05),
        Rotate(p=0.05),
        HFlip(p=0.05),
        VFlip(p=0.01),
        MotionBlur(p=0.05),
        GaussianBlur(p=0.01),
        Blur(p=0.01),
        Rain(p=0.05),
        Extend(p=0.01),
        BlockShuffle(p=0.1),
        LocalShuffle(p=0.01),
        RandomPadding(p=0.01),
        transforms.ToTensor()
    ])

    train(epochs, batch_size, transform, lr=lr, image_txt=image_txt, cut_mix=0.0)