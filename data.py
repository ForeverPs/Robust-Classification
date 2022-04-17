import os
import tqdm
import torch
from data_aug import *
from utils import parse_txt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split


options = [
    'Cassowary', 'Sparrow', 'Falcon', 'Owl', 'Duck', 'Cat', 'Lynx', 'Leopard', 'Lion', 'Tiger', 'Ladybird',
    'LongBeetle', 'ColoradoPotatoBeetle', 'DungBeetle', 'ElephantBeetle', 'Horse', 'Mule', 'Zebra', 'Yak',
    'Goat'
]

name2label = {options[i]: i for i in range(len(options))}


def get_val_pairs(val_txt):
    with open(val_txt, 'r') as f:
        lines = f.readlines()
    f.close()

    val_pairs = list()
    for line in lines:
        img_name, label = line.strip().split(' ')
        abs_img_name = 'data/track1_test1/%s' % img_name
        val_pairs.append((abs_img_name, int(label)))
    return val_pairs


class MyDataset(Dataset):
    def __init__(self, names, transform, path_prefix='data/train_phase1/images/'):
        self.names = names
        self.transform = transform
        self.prefix = path_prefix

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name, label = self.names[index]
        abs_img_name = '%s%s' % (self.prefix, img_name)
        img = Image.open(abs_img_name).convert('RGB')
        return self.transform(img), int(label)


def data_pipeline(image_txt, transform, batch_size, val_txt='data/test_label.txt'):
    # only center crop for validation image
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    data_pairs, _ = parse_txt(image_txt)
    val_pairs = get_val_pairs(val_txt)
    train_set = MyDataset(data_pairs, transform)
    val_set = MyDataset(val_pairs, val_transform, path_prefix='')
    all_set = ConcatDataset([train_set, val_set])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24)
    all_loader = DataLoader(all_set, batch_size=batch_size, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return all_loader, train_loader, val_loader


def val_data_pipeline(batch_size, val_txt='data/test_label.txt'):
    # only center crop for validation image
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    val_pairs = get_val_pairs(val_txt)
    val_set = MyDataset(val_pairs, val_transform, path_prefix='')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=24)
    return val_loader


if __name__ == '__main__':
    image_txt = 'data/train_phase1/label.txt'

    # data augmentation : just for testing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 64
    train_loader, val_loader = data_pipeline(image_txt, transform, batch_size)
    for x, y in tqdm.tqdm(val_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))