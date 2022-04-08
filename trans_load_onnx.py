import torch.onnx
# from model.resnet import resnet18
from model.bit_resnet import resnet18

x = torch.randn(1, 3, 224, 224, requires_grad=True)

def submit_model():
    model = resnet18(num_classes=20)  # 修改这里到指定模型
    return model

def load_state(path, model):
    try:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    except:
        new_dict = dict()
        for k, v in torch.load(path, map_location='cpu').items():
            new_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_dict, strict=True)
    return None

torch_model = submit_model()
load_path = 'log/checkpoint_bit_resnet_in_2022-04-04/epoch_276_acc_0.863.pth'  # 修改模型参数文件路径
load_state(load_path, torch_model)
torch_model.eval()

# load_state(load_path, torch_model)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "submit.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                #   opset_version=9,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])