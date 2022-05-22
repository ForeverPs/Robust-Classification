import tqdm
import json
import torch
import numpy as np
from PIL import Image
from model.convnext_cls import ConvNextCls
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from adv_gen import fgsm_attack, target_fgsm_attack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_acc(predict, target):
    predict = predict.detach().cpu().squeeze().numpy()
    target = target.detach().cpu().squeeze().numpy()
    acc = np.sum(predict == target) / len(predict)
    return acc


def get_name_label_pairs(json_name):
    with open(json_name, 'r') as f:
        jsc = json.load(f)
    
    name_label_pair = list()
    for k, v in jsc.items():
        label = int(k)
        images = v
        for image in images:
            abs_image_name = '/opt/tiger/debug_server/Phase2/data/train_p2/%s' % image
            name_label_pair.append((abs_image_name, label))
    return name_label_pair


class MyDataset(Dataset):
    def __init__(self, names, transform):
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name, label = self.names[index]
        img = Image.open(img_name).convert('RGB')
        return self.transform(img), int(label)


def get_val_loader(val_json, batch_size):
    # only center crop for validation image
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    val_pairs = get_name_label_pairs(val_json)
    val_set = MyDataset(val_pairs, val_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=24)
    return val_loader


def validation(batch_size=128):
    val_loader = get_val_loader(val_json, batch_size=batch_size)
    val_acc, target_val_acc, untarget_val_acc = 0, 0, 0
    
    for x, y in tqdm.tqdm(val_loader):
        x = x.float().to(device)
        y = y.long().to(device)

        # clean image
        predict = model(x)
        _, predict_cls = torch.max(predict, dim=-1)
        temp_acc = get_acc(predict_cls, y)
        val_acc += temp_acc

        # untarget fgsm attack
        untarget_x = fgsm_attack(model, x, y, epsilon=2/255)
        untarget_predict = model(untarget_x)
        _, untarget_predict_cls = torch.max(untarget_predict, dim=-1)
        temp_untarget_acc = get_acc(untarget_predict_cls, y)
        untarget_val_acc += temp_untarget_acc

        # target fgsm attack
        target_x = target_fgsm_attack(model, x)
        target_predict = model(target_x)
        _, target_predict_cls = torch.max(target_predict, dim=-1)
        temp_target_acc = get_acc(target_predict_cls, y)
        target_val_acc += temp_target_acc
        
    val_acc = val_acc / len(val_loader)
    untarget_val_acc = untarget_val_acc / len(val_loader)
    target_val_acc = target_val_acc / len(val_loader)
    print('Vanilla Acc: %.3f | Untarget Attack Acc: %.3f | Target Attack Acc: %.3f' % (val_acc, untarget_val_acc, target_val_acc))



if __name__ == '__main__':
    val_json = '/opt/tiger/debug_server/Phase2/data/val.json'
    model = ConvNextCls(num_classes=100)
    model_path = '/opt/tiger/debug_server/convnext_submit/baseline.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = model.eval().to(device)
    validation(batch_size=64)