import onnx
import onnxruntime
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pathlib
import numpy as np
import tqdm
import onnxruntime as ort

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


def test_pipeline(test_path='./data/track1_test2/', batch_size=1):
    # img_names = ['%s%s' % (test_path, img_name) for img_name in os.listdir(test_path)]
    test_set = MyDataset()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader



def get_numpy_acc(predict, target):
    predict = predict.reshape(-1)
    target = target.reshape(-1)
    acc = np.sum(predict == target) / len(predict)
    return acc



if __name__ == '__main__':

    onnx_model = onnx.load('submit.onnx')
    # onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession('submit.onnx', None)


    train_loss, val_loss, attack_loss = 0, 0, 0
    train_acc, val_acc, attack_acc = 0, 0, 0

    train_loader = test_pipeline()

    cnt = 0
    for x, y in tqdm.tqdm(train_loader):
        x = x.numpy()
        y = y.numpy()

        # print(x.shape)
        predict_cls = ort_sess.run(None, {'input': x})
        predict_cls = np.argmax(predict_cls, axis=1)
            
      
        # fgsm_x = attack.generate(x)
        # with torch.no_grad():
        #     predict_atk = classifier.predict(fgsm_x)
        # predict_atk = np.argmax(predict_atk, axis=1)


        
        # attack_acc += get_numpy_acc(predict_atk, y)
        train_acc += get_numpy_acc(predict_cls, y)  # * 0.9 + get_acc(predict_cls, y) * 0.1

        # if attack and cnt == 0:
        #     images = torch.concat([x[:8], fgsm_x[:8]], dim=0)
        #     save_image(images, f'test_{cnt}.png')
        # cnt += 1
            

       
    # train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    # attack_loss = attack_loss / len(train_loader)
    # attack_acc = attack_acc / len(train_loader)

    print('Clean Acc : %.3f ' % train_acc)






