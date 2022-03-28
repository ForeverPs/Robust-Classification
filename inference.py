import os
import json
import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model.se_resnet import se_resnet50, se_resnet34, se_resnet18


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):
    def __init__(self, names):
        self.names = names
        self.transform = infer_transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img = Image.open(img_name).convert('RGB')
        return self.transform(img), img_name


def test_pipeline(test_path, batch_size):
    img_names = ['%s%s' % (test_path, img_name) for img_name in os.listdir(test_path)]
    test_set = MyDataset(img_names)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader


def infer(test_path, batch_size, model_path):
    test_loader = test_pipeline(test_path, batch_size)
    # model = se_resnet34(num_classes=20)
    model = se_resnet18(num_classes=20)

    # remove module prefix
    new_dict = dict()
    for k, v in torch.load(model_path, map_location='cpu').items():
        new_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_dict, strict=True)
    model = model.to(device).eval()
    print('Successfully load trained model...')

    img_id = list()
    result_label = list()
    with torch.no_grad():
        for x, name in tqdm.tqdm(test_loader):
            x = x.float().to(device)
            y = model(x)
            conf, label = torch.max(y, dim=-1)
            label = label.detach().cpu().numpy().tolist()
            result_label.extend(label)
            img_id.extend(list(name))

    assert len(img_id) == len(result_label)
    json_list = list()
    for image_id, category_id in zip(img_id, result_label):
        img_id = image_id.split('/')[-1].replace('.png', '')
        json_list.append({'image_id': int(img_id), 'category_id': int(category_id)})

    with open('t1_p1_result.json', 'w') as f:
        json.dump(json_list, f)

    print('Inference Finished on %d Images.' % len(json_list))


if __name__ == '__main__':
    test_path = 'data/track1_test1/'
    batch_size = 64
    model_path = 'saved_models/epoch_400_acc_1.000.pth'
    infer(test_path, batch_size, model_path)

