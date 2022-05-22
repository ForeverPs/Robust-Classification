from data_aug import *
import torchvision.transforms as transforms


# def get_full_transform():
#     # data augmentation
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomResizedCrop(224), 
#         PepperSaltNoise(p=0.1),
#         ColorPointNoise(p=0.1),
#         GaussianNoise(p=0.1),
#         Mosaic(p=0.1),
#         RGBShuffle(p=0.1),
#         Rotate(p=0.1),
#         HFlip(p=0.1),
#         VFlip(p=0.01),
#         GaussianBlur(p=0.001),
#         Blur(p=0.001),
#         Rain(p=0.01),
#         Extend(p=0.01),
#         BlockShuffle(p=0.01),
#         LocalShuffle(p=0.05),
#         RandomPadding(p=0.1),
#         Fog(p=0.01),
#         ShotNoise(p=0.1),
#         ImpulseNoise(p=0.1),
#         SpeckleNoise(p=0.1),
#         GlassBlur(p=0.01),
#         DeFocusBlur(p=0.01),
#         MotionBlur(p=0.01),
#         ZoomBlur(p=0.01),
#         Frost(p=0.1),
#         Snow(p=0.1),
#         Spatter(p=0.05),
#         Contrast(p=0.05),
#         Brightness(p=0.05),
#         Saturate(p=0.05),
#         Compress(p=0.05),
#         Pixelate(p=0.05),
#         Elastic(p=0.05),
#         transforms.ToTensor()
#     ])
#     return transform


def get_full_transform(p):
    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224), 
        PepperSaltNoise(p=p),
        ColorPointNoise(p=p),
        GaussianNoise(p=p),
        Mosaic(p=p),
        RGBShuffle(p=p),
        Rotate(p=p),
        HFlip(p=p),
        VFlip(p=p),
        GaussianBlur(p=p),
        Blur(p=p),
        Rain(p=p),
        Extend(p=p),
        BlockShuffle(p=p),
        LocalShuffle(p=p),
        RandomPadding(p=p),
        Fog(p=p),
        ShotNoise(p=p),
        ImpulseNoise(p=p),
        SpeckleNoise(p=p),
        GlassBlur(p=p),
        DeFocusBlur(p=p),
        MotionBlur(p=p),
        ZoomBlur(p=p),
        Frost(p=p),
        Snow(p=p),
        Spatter(p=p),
        Contrast(p=p),
        Brightness(p=p),
        Saturate(p=p),
        Compress(p=p),
        Pixelate(p=p),
        Elastic(p=p),
        transforms.ToTensor()
    ])
    return transform