import tqdm
import torch
from data_aug import *
from utils import parse_txt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


class MyDataset(Dataset):
    def __init__(self, names, transform, path_prefix='train_phase1/images/'):
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


def data_pipeline(image_txt, transform, batch_size, val_ratio=0.2):
    data_pairs, _ = parse_txt(image_txt)
    val_size = int(len(data_pairs) * val_ratio)
    train_size = len(data_pairs) - val_size
    dataset = MyDataset(data_pairs, transform)
    # fixed split
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader


if __name__ == '__main__':
    image_txt = 'train_phase1/label.txt'

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 64
    val_ratio = 0.2
    train_loader, val_loader = data_pipeline(image_txt, transform, batch_size, val_ratio)
    for x, y in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))