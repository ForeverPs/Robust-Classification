import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from utils import motion_blur, rain_noise, extend, local_shuffle


class PepperSaltNoise:
    def __init__(self, p=.3, snr=.99):
        # snr : signal / noise
        # p : probability of data augmentation
        self.snr = snr
        self.p = p
        self.noise_degree = 1 - snr

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # select channel randomly
            mask = np.random.choice(list(range(c)), size=(h, w, 1),
                                    p=[self.snr, self.noise_degree / 2., self.noise_degree / 2.])
            mask = np.repeat(mask, c, axis=2)
            # channel 0 : unchanged
            # channel 1 : salt noise
            # channel 2 : pepper noise
            img_[mask == 1] = 255  # salt noise
            img_[mask == 2] = 0    # pepper noise
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class ColorPointNoise:
    def __init__(self, p=.3, snr=.99):
        # snr : signal / noise
        # p : probability of data augmentation
        self.snr = snr
        self.p = p
        self.noise_degree = 1 - snr

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # select channel randomly
            mask = np.random.choice(list(range(c + 2)), size=(h, w),
                                    p=[self.noise_degree / 4., self.noise_degree / 4.,
                                       self.noise_degree / 4., self.noise_degree / 4., self.snr])
            # mask 0 : green noise
            # mask 1 : red noise
            # mask 2 : blue noise
            # mask 3 : random noise
            # mask 4 : unchanged
            img_[mask == 0] = np.array([0, 255, 0]).reshape((1, 3))  # green noise
            img_[mask == 1] = np.array([255, 0, 0]).reshape((1, 3))  # red noise
            img_[mask == 2] = np.array([0, 0, 255]).reshape((1, 3))  # blue noise
            img_[mask == 3] = (255 * np.random.uniform(0, 1, size=img_[mask == 3].shape)).astype(np.uint8)  # color noise
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class GaussianNoise:
    def __init__(self, p=.3, mean=0, std=0.15):
        # p : probability of data augmentation
        self.p = p
        self.mean = mean
        self.std = np.random.choice(np.linspace(0, std, 30), size=1)[0]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            gaussian = np.random.normal(self.mean, self.std, size=img_.shape)
            img_ = img_ / 255.0 + gaussian
            img_ = 255 * np.clip(img_, 0, 1)
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class Mosaic:
    def __init__(self, p=.3, scale=None):
        # p : probability of data augmentation
        if scale is None:
            scale = list(range(13, 27))
        self.p = p
        self.scale = scale
        self.mosaic_type = [0, 1, 2, 3]
        self.mosaic_p = [0.7, 0.1, 0.1, 0.1]
        # 0.7 -> 0 : color mosaic
        # 0.1 -> 1 : black mosaic
        # 0.1 -> 2 : white mosaic
        # 0.1 -> 3 : gray mosaic

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            scale = np.random.choice(self.scale, size=1)[0]
            h_ = np.random.choice(list(range(2 * scale, h - 2 * scale)), size=1)[0]
            w_ = np.random.choice(list(range(2 * scale, w - 2 * scale)), size=1)[0]
            mosaic_type = np.random.choice(self.mosaic_type, size=1, p=self.mosaic_p)[0]
            if mosaic_type == 0:
                mosaic = np.random.uniform(0, 255, size=(scale, scale, 3)).astype(np.uint8)
            elif mosaic_type == 1:
                mosaic = np.zeros((scale, scale, 3)).astype(np.uint8)
            elif mosaic_type == 2:
                mosaic = 255 * np.ones((scale, scale, 3)).astype(np.uint8)
            else:
                mosaic = 127 * np.ones((scale, scale, 3)).astype(np.uint8)
            img_[h_: h_ + scale, w_: w_ + scale, :] = mosaic
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class RGBShuffle:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            channels = list(range(c))
            np.random.shuffle(channels)
            img_[..., np.array(list(range(c)))] = img_[..., np.array(channels)]
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class Rotate:
    def __init__(self, p=.3, angle=5):
        # p : probability of data augmentation
        self.p = p
        self.angle = [-1 * angle, angle]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = transforms.RandomRotation(self.angle)(img)
            return img_
        return img


class HFlip:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        return transforms.RandomHorizontalFlip(p=self.p)(img)


class VFlip:
    def __init__(self, p=.1):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        return transforms.RandomVerticalFlip(p=self.p)(img)


class RandomCut:
    def __init__(self, p=.3, scale=None):
        # p : probability for cutting
        self.p = p
        if scale is None:
            scale = [0.1 * i for i in range(4, 10)]
        self.scale = scale

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            scale = np.random.choice(self.scale, size=1)[0]
            h_ = np.random.choice(list(range(h - int(scale * h) - 1)), size=1)[0]
            w_ = np.random.choice(list(range(w - int(scale * w) - 1)), size=1)[0]
            img_ = img_[h_: h_ + int(scale * h), w_: w_ + int(w * scale)]
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class MotionBlur:
    def __init__(self, p=.3, degree=15, angle=90):
        # p : probability of data augmentation
        self.p = p
        self.degree = np.random.choice(list(range(5, degree)), size=1)[0]
        self.angle = np.random.choice(list(range(20, angle)), size=1)[0]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            img_ = motion_blur(img_, self.degree, int(self.angle))
            img_ = Image.fromarray(img_).convert('RGB')
            return img_
        return img


class GaussianBlur:
    def __init__(self, p=.3, radius=2):
        # p : probability of data augmentation
        self.p = p
        self.radius = np.random.choice(list(range(1, radius + 1)), size=1)[0]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img


class Blur:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img = img.filter(ImageFilter.BLUR)
        return img


class Rain:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p
        self.value = np.random.choice([500, 700, 900, 1100, 1300, 1500], size=1)[0]
        self.angle = np.random.choice(np.linspace(-30, 30, 10), size=1)[0]
        self.beta = np.random.choice(np.linspace(0.7, 1, 10), size=1)[0]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            img_ = rain_noise(img_, value=self.value, angle=self.angle, beta=self.beta)
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class Extend:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p
        self.offset = np.random.choice(list(range(3, 10)), size=1)[0]

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            img_ = extend(img_, self.offset)
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class BlockShuffle:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img_ = np.zeros(img.shape)
            h, w, _ = img_.shape
            style = np.random.choice([0, 1], size=1)[0]
            if style == 0:  # split vertical
                h_split = int(np.random.choice(np.linspace(0.4 * h, 0.6 * h, num=10), size=1)[0])
                img_[:h - h_split] = img[h_split:]
                img_[h - h_split:] = img[:h_split]
            else:  # split horizontal
                w_split = int(np.random.choice(np.linspace(0.4 * w, 0.6 * w, num=10), size=1)[0])
                img_[:, :w - w_split, :] = img[:, w_split:, :]
                img_[:, w - w_split:, :] = img[:, :w_split, :]
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class LocalShuffle:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p
        self.scale = int(np.random.choice(np.linspace(80, 100, num=20), size=1)[0])

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            img_ = local_shuffle(img_, patch_size=(self.scale, self.scale))
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img


class RandomPadding:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            w, h = img.size
            scale = np.random.choice(np.linspace(0.65, 0.95, num=10), size=1)[0]
            w_, h_ = int(w * scale), int(h * scale)
            resize_img = np.array(img.resize((w_, h_)))
            style = np.random.choice([0, 1], size=1)[0]
            if style == 0:  # gray padding
                img_ = 255 * np.random.choice(np.linspace(0, 1, num=10), size=1)[0] * np.ones((h, w, 3))
            else:  # color padding
                img_ = 255 * np.random.uniform(0, 1, size=(h, w, 3))
            start_w = int(np.random.choice(np.linspace(0, int(w * (1 - scale)) - 1, num=10), size=1)[0])
            start_h = int(np.random.choice(np.linspace(0, int(h * (1 - scale)) - 1, num=10), size=1)[0])
            img_[start_h: start_h + h_, start_w: start_w + w_, :] = resize_img
            return Image.fromarray(img_.astype(np.uint8)).convert('RGB')
        return img

if __name__ == '__main__':
    img_name = 'data_aug_test/test.JPEG'
    img = Image.open(img_name)
    # img = PepperSaltNoise(p=1)(img)
    # img = ColorPointNoise(p=0.2)(img)
    # img = GaussianNoise(p=0.2)(img)
    # img = Mosaic(p=0.2)(img)
    # img = RGBShuffle(p=0.05)(img)
    # img = Rotate(p=0.1, angle=10)(img)
    # img = HFlip(p=0.1)(img)
    # img = VFlip(p=0.05)(img)
    # img = RandomCut(p=0.2)(img)
    # img = MotionBlur(p=0.1)(img)
    # img = GaussianBlur(p=0.01)(img)
    # img = Blur(p=0.05)(img)
    # img = Rain(p=0.2)(img)
    img = BlockShuffle(p=1)(img)
    plt.imshow(img)
    plt.show()
    # plt.imsave('img.png', np.array(img))




