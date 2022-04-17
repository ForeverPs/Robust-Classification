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
from data_transform import get_full_transform
from model.se_resnet import ResizedPadSeResNetML
from adv_gen import fgsm_attack, target_fgsm_attack
from utils import cutmix, mixup_data, mixup_criterion

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, transform, lr=1e-3, image_txt='data/train_phase1/label.txt', cut_mix=0.0):
    all_loader, train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)
    model = ResizedPadSeResNetML(depth=18, num_classes=20, l=218)

    try:
        model.load_state_dict(torch.load(model_path), strict=True)
        print('Training with a Trained Model...')
    except:
        print('Training from Scratch...')

    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.86
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc, target_acc, untarget_acc = 0, 0, 0, 0

        model.train()
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

            # TRADES
            loss = loss_ + loss_mixup + adv_weight * loss_adv_white + adv_weight * loss_adv_target_white

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            _, predict_cls_adv_white = torch.max(predict_adv_white, dim=-1)
            _, predict_cls_adv_target_white = torch.max(predict_adv_target_white, dim=-1)
            train_acc += ((get_acc(predict_cls, y) + get_acc(predict_cls_adv_target_white, double_y) * adv_weight + get_acc(predict_cls_adv_white, double_y) * adv_weight) / (1 + 2 * adv_weight))

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

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)

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
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.state_dict(), '%s/aug_adv_%s.pth' % (save_path, model_name))

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/target_acc', target_acc, epoch)
        writer.add_scalar('val/untarget_acc', untarget_acc, epoch)


if __name__ == '__main__':
    model_path = None
    logdir = './tensorboard/ResizePad18_allset_no_black_box/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir=logdir)

    save_path = './saved_models/ResizePad18_allset_no_black_box'

    lr = 1e-3
    epochs = 3000
    batch_size = 128
    adv_weight = 1  # TRADES : 1 or 6
    image_txt = 'data/train_phase1/label.txt'
    transform = get_full_transform()
    train(epochs, batch_size, transform, lr=lr, image_txt=image_txt, cut_mix=0.3)