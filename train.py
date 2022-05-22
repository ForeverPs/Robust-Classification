import os
import tqdm
import torch
import shutil
import argparse
from torch import optim
from eval import get_acc
from torch import nn, optim
import torch.distributed as dist
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.convnext_cls import ConvNextCls
from data_transform import get_full_transform
from data import get_train_dataset, get_val_dataset
from utils import cutmix, mixup_data, mixup_criterion

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'


def main(opt):
    if opt.local_rank == 0 and opt.build_tensorboard:
        shutil.rmtree(opt.logdir, True)
        writer = SummaryWriter(logdir=opt.logdir)
        opt.build_tensorboard = False
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method, world_size=opt.n_gpus)

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    train_set = get_train_dataset(opt.train_json, transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=36)

    val_set = get_val_dataset(opt.val_json)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=36)
        
    model = ConvNextCls(num_classes=100)

    params = [
    {'params': model.cls_head.parameters(), 'lr': opt.cls_head_lr},
    {'params': model.backbone.parameters(), 'lr': opt.backbone_lr},
    ]

    try:
        model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
        print('Load my pretrained models')
    except:
        print('Training from scratch...')

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)

    optimizer = optim.Adamax(params, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(opt.epoch):
        train_loader.sampler.set_epoch(epoch) 

        if opt.local_rank == 0:
            data_loader = tqdm.tqdm(train_loader)
        else:
            data_loader = train_loader
        
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        model.train()
        for x, y in data_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            predict = model(x)
            vanilla_loss = criterion(predict, y)

            # Mix Up training
            inputs, targets_a, targets_b, lam = mixup_data(x, y, 1.0)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            mix_up_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # # Cut Mix training
            cut_mix_x, target_a, target_b, lam = cutmix(x, y)
            cut_mix_predict = model(cut_mix_x)
            cut_mix_loss = criterion(cut_mix_predict, target_a) * lam + criterion(cut_mix_predict, target_b) * (1. - lam)
            
            loss = vanilla_loss + cut_mix_loss + mix_up_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predict_cls = torch.max(predict, dim=-1)
            train_acc += get_acc(predict_cls, y)

        # update learning rate
        scheduler.step()

        if opt.local_rank == 0 and epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                for x, y in tqdm.tqdm(val_loader):
                    x = x.float().to(device)
                    y = y.long().to(device)

                    # original image
                    predict = model(x)
                    loss = criterion(predict, y)

                    val_loss += loss.item()

                    _, predict_cls = torch.max(predict, dim=-1)
                    val_acc += get_acc(predict_cls, y)

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)

            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_loader)

            print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | Val Acc : %.3f'
                % (epoch, train_loss, train_acc, val_loss, val_acc))

            if val_acc >= opt.best_acc:
                opt.best_acc = val_acc
                model_name = 'epoch_%d_acc_%.3f.pth' % (epoch, val_acc)
                os.makedirs(opt.save_path, exist_ok=True)
                torch.save(model.module.state_dict(), '%s/%s' % (opt.save_path, model_name))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    # data augmentation
    transform = get_full_transform(p=0.05)

    parser = argparse.ArgumentParser('Phase2 training script.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--cls_head_lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank of current process')
    parser.add_argument('--init_method', default='env://')
    parser.add_argument('--n_gpus', type=int, default=5)
    parser.add_argument('--build_tensorboard', type=bool, default=True)
    parser.add_argument('--train_json', type=str, default='./data/train.json')
    parser.add_argument('--val_json', type=str, default='./data/val.json')
    parser.add_argument('--logdir', type=str, default='./tensorboard/Phase2_convnext_tiny')
    parser.add_argument('--save_path', type=str, default='./saved_models/convnext_tiny')
    parser.add_argument('--best_acc', type=float, default=0.9)
    parser.add_argument('--model_path', type=str, default='./saved_models/checkpoint_epoch_62.pth')

    opt = parser.parse_args()
    if opt.local_rank == 0:
        print('opt:', opt)

    main(opt)

# python -m torch.distributed.launch --nproc_per_node=5 train.py  --batch_size 64 --n_gpus=5