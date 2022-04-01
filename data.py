import os
import tqdm
import torch
from data_aug import *
from utils import parse_txt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


options = [
    'Cassowary', 'Sparrow', 'Falcon', 'Owl', 'Duck', 'Cat', 'Lynx', 'Leopard', 'Lion', 'Tiger', 'Ladybird',
    'LongBeetle', 'ColoradoPotatoBeetle', 'DungBeetle', 'ElephantBeetle', 'Horse', 'Mule', 'Zebra', 'Yak',
    'Goat'
]

name2label = {options[i]: i for i in range(len(options))}


def get_val_pairs(val_path):
    val_pairs = list()
    for folder in options:
        label = name2label[folder]
        img_path = '%s%s' % (val_path, folder)
        for img_name in os.listdir(img_path):
            abs_img_name = '%s/%s' % (img_path, img_name)
            val_pairs.append((abs_img_name, label))
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


def data_pipeline(train_image_txt, val_image_text, transform, batch_size):
    # only center crop for validation image
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    data_pairs, _ = parse_txt(train_image_txt)
    val_pairs, _ = parse_txt(val_image_text)
    train_set = MyDataset(data_pairs, transform)
    val_set = MyDataset(val_pairs, val_transform, path_prefix='data/track1_test1/images/')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, val_loader


if __name__ == '__main__':
    image_txt = 'data/train_phase1/label.txt'
    val_image_txt = 'data/track1_test1/label.txt'

    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        PepperSaltNoise(p=0.15),
        ColorPointNoise(p=0.2),
        GaussianNoise(p=0.2),
        Mosaic(p=0.2),
        RGBShuffle(p=0.05),
        Rotate(p=0.15),
        HFlip(p=0.1),
        VFlip(p=0.05),
        RandomCut(p=0.1),
        MotionBlur(p=0.15),
        GaussianBlur(p=0.01),
        Blur(p=0.03),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 64
    val_ratio = 0.2
    train_loader, val_loader = data_pipeline(image_txt, val_image_txt, transform, batch_size)
    for x, y in tqdm.tqdm(val_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))
