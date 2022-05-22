import tqdm
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


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


def data_pipeline(train_json, val_json, transform, batch_size):
    # only center crop for validation image
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    train_pairs = get_name_label_pairs(train_json)
    val_pairs = get_name_label_pairs(val_json)
    train_set = MyDataset(train_pairs, transform)
    val_set = MyDataset(val_pairs, val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, val_loader


def get_train_dataset(train_json, transform):
    train_pairs = get_name_label_pairs(train_json)
    train_set = MyDataset(train_pairs, transform)
    return train_set


def get_val_dataset(val_json):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    val_pairs = get_name_label_pairs(val_json)
    val_set = MyDataset(val_pairs, val_transform)
    return val_set



if __name__ == '__main__':
    train_json = './data/train.json'
    val_json = './data/val.json'

    # data augmentation : just for testing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 64
    train_loader, val_loader = data_pipeline(train_json, val_json, transform, batch_size)
    for x, y in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))
    