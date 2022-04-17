import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# # mean kernel
# def mean_kernel(kernel_size):
#     w = torch.ones((kernel_size, kernel_size))
#     w = w / torch.sum(w)
#     return w


# # gaussian blur kernel
# def gaussian_kernel(kernel_size, sigma=1):
#     x = torch.arange(kernel_size) - kernel_size // 2
#     gauss = torch.exp(-x.pow(2.0)) / (2 * sigma ** 2)
#     gauss = gauss / torch.sum(gauss)
#     gauss = gauss.unsqueeze(1) * gauss.unsqueeze(0)
#     return gauss


# # smooth auxiliary function
# class SmoothFilter(nn.Module):
#     def __init__(self,
#                  filter='mean',
#                  in_channel=3,
#                  kernel_size=5,
#                  niter=1,
#                  sigma=2.,
#                  ):
#         super().__init__()
#         if filter == 'mean':
#             w = mean_kernel(kernel_size)
#         if filter == 'gauss':
#             w = gaussian_kernel(kernel_size, sigma)

#         w = w.repeat((in_channel, 1, 1, 1))
#         self.register_buffer('w', w)

#         self.padding = kernel_size // 2
#         self.in_channel = in_channel
#         self.n_iter = niter

#     def forward(self, x):
#         with torch.no_grad():
#             for _ in range(self.n_iter):
#                 x = F.conv2d(x, self.w.to(x.device), padding=self.padding, groups=self.in_channel)
#         return x


# # smooth operation
# class Smoothflow(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, filter, in_channel, kernel_size, niter, sigma):
#         input = SmoothFilter(filter=filter, in_channel=in_channel,
#                                 kernel_size=kernel_size, niter=niter,
#                                 sigma=sigma)(input)
#         return input.clamp(0, 1)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # no change the grad
#         # None * 5 is the gradient of filter, in_channel, kernel_size, niter, sigma
#         return grad_output, None, None, None, None, None


# class SmoothLayer(nn.Module):
#     def __init__(self, filter='gauss', in_channel=3, kernel_size=3, niter=1, sigma=3):
#         super(SmoothLayer, self).__init__()
#         self.filter = filter
#         self.in_channel = in_channel
#         self.kernel_size = kernel_size
#         self.niter = niter
#         self.sigma = sigma

#     def forward(self, x):
#         out = Smoothflow.apply(x, self.filter, self.in_channel,
#                                self.kernel_size, self.niter,
#                                self.sigma)
#         return out


# # bit operation
# class Bitflow(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, b):
#         # scale = 1 / 2 ** b
#         # out = torch.quantize_per_tensor(x, scale=scale, zero_point=0, dtype=torch.quint8).dequantize()
#         scale = 2 ** b
#         out = torch.div(x, 1 / scale, rounding_mode='trunc') / scale
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         # None is the gradient of b

#         # during inference
#         # eps = 1e-6
#         # noise = torch.randn(grad_output.shape).to(grad_output.device)
#         # grad_output = torch.sign(noise) * grad_output * eps
#         # grad_output = -grad_output * eps
#         return grad_output, None


# # bit layer
# class BitLayer(nn.Module):
#     def __init__(self, bit):
#         super(BitLayer, self).__init__()
#         self.bit = bit

#     def forward(self, x):
#         out = Bitflow.apply(x, self.bit)
#         return out


# class RobustModule(nn.Module):
#     def __init__(self, filter='mean', in_channel=3,
#                  kernel_size=5, niter=1, sigma=3., bit=6):
#         super().__init__()
#         self.smooth = SmoothLayer(filter, in_channel, kernel_size, niter, sigma)
#         self.tobit = BitLayer(bit)

#     def forward(self, x):
#         x = self.smooth(x)
#         x = self.tobit(x)
#         return x


# For phase 1
# class ResizePaddingFlow(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, l):
#         # x : batch, 3, 224, 224
#         with  torch.no_grad():
#             pad1 = int((224 - l) / 2)
#             pad2 = 224 - l - pad1
#             resize = F.interpolate(x, (l, l))
#             padding = nn.ZeroPad2d(padding=(pad1, pad2, pad1, pad2))(resize)
#         return padding

#     @staticmethod
#     def backward(ctx, grad_output):
#         # None: gradients of l
#         return grad_output, None


# class ResizedPaddingLayer(nn.Module):
#     def __init__(self, l):
#         super(ResizedPaddingLayer, self).__init__()
#         self.l = l

#     def forward(self, x):
#         out = ResizePaddingFlow.apply(x, self.l)
#         return out


class ResizedPaddingLayer(nn.Module):
    def __init__(self, l):
        super(ResizedPaddingLayer, self).__init__()
        self.l = l
        pad1 = int((224 - l) / 2)
        pad2 = 224 - l - pad1
        self.padding = nn.ZeroPad2d(padding=(pad1, pad2, pad1, pad2))

    def forward(self, x):
        resize = F.interpolate(x, (self.l, self.l))
        resize = self.padding(resize)
        return resize


if __name__ == '__main__':
    x = torch.rand(10, 3, 224, 224)
    random_layer = ResizedPaddingLayer(l=214)
    out = random_layer(x)
    print(out.shape, torch.min(out), torch.max(out))