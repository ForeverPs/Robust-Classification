import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms


def mean_kernel(kernel_size):
    w = torch.ones((kernel_size, kernel_size))
    w = w / torch.sum(w)
    return w

def gaussian_kernel(kernel_size, sigma=1):
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x.pow(2.0)) / (2 * sigma ** 2)
    gauss = gauss / torch.sum(gauss)
    gauss = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    return gauss


class SmoothFilter(nn.Module):
    def __init__(self, 
        filter='mean', 
        in_channel=3,
        kernel_size=5, 
        niter=1,
        sigma=2.,
    ):
        super().__init__()
        if filter == 'mean':
            w = mean_kernel(kernel_size)
        if filter == 'gauss':
            w = gaussian_kernel(kernel_size, sigma)
        
        w = w.repeat((in_channel, 1, 1, 1))
        self.register_buffer('w', w)

        self.padding = kernel_size // 2
        self.in_channel = in_channel
        self.n_iter = niter

    
    def forward(self, x):
        for _ in range(self.n_iter):
            x = F.conv2d(x, self.w, padding=self.padding, groups=self.in_channel)
        
        return x


class Bitflow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_):
        scale = 1 / 2 ** b_
        out = torch.quantize_per_tensor(
                x, 
                scale=scale, 
                zero_point=0,
                dtype=torch.quint8,
            ).dequantize()
        # out.requires_grad = x.requires_grad
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# class Bitflow(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, b_):
#         scale = 1 / 2 ** b_
#         out = torch.quantize_per_tensor(
#                 x, 
#                 scale=scale, 
#                 zero_point=0,
#                 dtype=torch.quint8,
#             ).dequantize()
#         out.requires_grad = x.requires_grad
#         return out
    
#     def backward(self, grad_output):
#         return grad_output


class SmoothQuantilization(nn.Module):
    def __init__(
        self, 
        filter='mean', 
        in_channel=3,
        kernel_size=5, 
        niter=1,
        sigma=3.,
        bit=6,
    ):
        super().__init__()
        self.smooth = SmoothFilter(filter, in_channel, kernel_size, niter, sigma)
        # self.tobit = Bitflow()
        self.nbit = bit
    
    def forward(self, x):
        x = self.smooth(x)
        x = Bitflow.apply(x, self.nbit)
        return x



if __name__ == '__main__':
    img = Image.open('./data/train_phase1/images/0.JPEG')
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    print(tensor.shape)

    smooth = SmoothQuantilization('gauss', 3, 5, 10, sigma=5, bit=6)
    tensor = smooth(tensor)
    print(tensor.shape)
    tensor = tensor.squeeze(0)
    print(torch.max(tensor))
    img = transforms.ToPILImage()(tensor)
    print(img)
    img.save('blur.png')

    


