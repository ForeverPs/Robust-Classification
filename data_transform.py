from data_aug import *
import torchvision.transforms as transforms


def get_full_transform():
    # data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224), 
        PepperSaltNoise(p=0.1),
        ColorPointNoise(p=0.1),
        GaussianNoise(p=0.1),
        Mosaic(p=0.1),
        RGBShuffle(p=0.05),
        Rotate(p=0.05),
        HFlip(p=0.1),
        VFlip(p=0.01),
        GaussianBlur(p=0.001),
        Blur(p=0.001),
        Rain(p=0.05),
        Extend(p=0.01),
        BlockShuffle(p=0.05),
        LocalShuffle(p=0.01),
        RandomPadding(p=0.01),
        Fog(p=0.05),
        ShotNoise(p=0.1),
        ImpulseNoise(p=0.1),
        SpeckleNoise(p=0.1),
        GlassBlur(p=0.1),
        DeFocusBlur(p=0.1),
        MotionBlur(p=0.1),
        ZoomBlur(p=0.1),
        # Frost(p=1),
        Snow(p=0.1),
        Spatter(p=0.1),
        Contrast(p=0.05),
        Brightness(p=0.05),
        Saturate(p=0.05),
        Compress(p=0.01),
        Pixelate(p=0.1),
        Elastic(p=0.1),
        transforms.ToTensor()
    ])
    return transform