import os
import tqdm
import torch
import torch.nn as nn
import numpy as np

import pathlib
from data import data_pipeline
# from model.se_resnet import se_resnet50, se_resnet18, se_resnet34
from torchvision.utils import save_image
from model.se_resnet import ResizedPadSeResNetML
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch, CarliniL2Method
from art.estimators.classification import PyTorchClassifier


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class MyDataset(Dataset):
    def __init__(self, path='./data/track1_test2/'):
        self.names = list((pathlib.Path(path) / 'images').glob('*'))
        # self.names = [str(x) for x in self.names]
        self.names = sorted(self.names)

        self.labels = dict()
        with open(path + 'labels.txt', 'r') as f:
            while True:
                try:
                    name, label = f.readline().strip().split(' ')
                    # print(name)
                    self.labels[name] = int(label)
                except:
                    break
        self.transform = infer_transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img = Image.open(str(img_name)).convert('RGB')
        label = self.labels[img_name.name]
        return self.transform(img), label


def test_pipeline(test_path, batch_size=64):
    # img_names = ['%s%s' % (test_path, img_name) for img_name in os.listdir(test_path)]
    test_set = MyDataset()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader



def get_numpy_acc(predict, target):
    predict = predict.reshape(-1)
    target = target.reshape(-1)
    acc = np.sum(predict == target) / len(predict)
    return acc


def attack(
    batch_size,
    train_image_txt='data/train_phase1/label.txt',
    val_image_txt='data/track1_test1/label.txt',
    checkpoint_path='',
    attack=True,
):
    train_loader = test_pipeline('./data/')

    # model = se_resnet18(num_classes=20)
    model = ResizedPadSeResNetML(depth=18, num_classes=20, l=218, use_robust=True).eval()

    
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("with out module. loading ...")
    except:
        new_dict = dict()
        for k, v in torch.load(checkpoint_path, map_location='cpu').items():
            new_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_dict, strict=True)
        print('with module. loading ...')


    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model, 
        clip_values=(0, 1), 
        loss=criterion, 
        input_shape=(3, 224, 224),
        nb_classes=20
    )

   
    criterion = nn.CrossEntropyLoss()

    # attack = FastGradientMethod(estimator=classifier, eps=8/255)

     # pgd
    # norm np.inf, 1, 2
    attack = ProjectedGradientDescentPyTorch(
        estimator=classifier, 
        norm=np.inf, 
        eps=8/255, 
        eps_step=8/255/40*3, 
        max_iter=20
    )
    # attack_str = 'pgd_2_255_2'

    # cw2
    # attack = CarliniL2Method(
    #     classifier=classifier, 
    #     binary_search_steps=5, 
    #     max_iter=10, 
    #     learning_rate=0.01,
    # )
    # attack_str = 'cw2'

   
    train_loss, val_loss, attack_loss = 0, 0, 0
    train_acc, val_acc, attack_acc = 0, 0, 0

    cnt = 0
    for x, y in tqdm.tqdm(train_loader):
        x = x.numpy()
        y = y.numpy()

        with torch.no_grad():
            predict_cls = classifier.predict(x)
            predict_cls = np.argmax(predict_cls, axis=1)
            
      
        fgsm_x = attack.generate(x)
        with torch.no_grad():
            predict_atk = classifier.predict(fgsm_x)
        predict_atk = np.argmax(predict_atk, axis=1)


        
        attack_acc += get_numpy_acc(predict_atk, y)
        train_acc += get_numpy_acc(predict_cls, y)  # * 0.9 + get_acc(predict_cls, y) * 0.1

        # if attack and cnt == 0:
        #     images = torch.concat([x[:8], fgsm_x[:8]], dim=0)
        #     save_image(images, f'test_{cnt}.png')
        # cnt += 1
            

       
    # train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    # attack_loss = attack_loss / len(train_loader)
    attack_acc = attack_acc / len(train_loader)

    print('Clean Acc : %.3f | Atk Acc : %.3f' % (train_acc, attack_acc))

       


if __name__ == '__main__':
    
    batch_size = 256
    train_image_txt = 'data/train_phase1/label.txt'
    val_image_txt = 'data/track1_test1/label.txt'
    checkpoint_path = 'phase1_model.pth'

    attack(
        batch_size, 
        train_image_txt=train_image_txt,
         val_image_txt=val_image_txt, 
         checkpoint_path=checkpoint_path,
         attack=True,
         )
