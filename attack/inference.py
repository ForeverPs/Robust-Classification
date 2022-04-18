import torch
from model.se_resnet import SmoothBitSeResNetML


if __name__ == '__main__':
    model_path = 'epoch_9_acc_0.969.pth'
    model = SmoothBitSeResNetML(depth=18, num_classes=20, num_bit=6).eval()

    try:
        # remove module prefix
        new_dict = dict()
        for k, v in torch.load(model_path, map_location='cpu').items():
            new_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_dict, strict=True)
        print('Successfully Loading Trained Model...')
    except:
        print('No Trained Model...')

    x = torch.randn(64, 3, 224, 224)
    y = model(x)
    print(y.shape)