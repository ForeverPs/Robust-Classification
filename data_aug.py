import cv2
import math
import random
import numpy as np
from utils import *
import skimage as sk
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from pkg_resources import resource_filename
import torchvision.transforms as transforms


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


# class MotionBlur:
#     def __init__(self, p=.3, degree=15, angle=90):
#         # p : probability of data augmentation
#         self.p = p
#         self.degree = np.random.choice(list(range(5, degree)), size=1)[0]
#         self.angle = np.random.choice(list(range(20, angle)), size=1)[0]
#
#     def __call__(self, img):
#         # img : PIL image object
#         if random.uniform(0, 1) < self.p:
#             img_ = np.array(img).copy()
#             img_ = motion_blur(img_, self.degree, int(self.angle))
#             img_ = Image.fromarray(img_).convert('RGB')
#             return img_
#         return img


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


class Fog:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def fog(self, x, severity=1):
        # x : PIL Image, 0-255
        x = np.array(x) / 255.0
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
        max_val = x.max()

        mapsize = 2 ** math.ceil(np.log2(max(x.shape[0], x.shape[1])))
        fog_mask = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])
        x += c[0] * fog_mask[:x.shape[0], :x.shape[1]][..., np.newaxis]
        return np.clip(x * max_val / (max_val + c[0]), 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.fog(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class ShotNoise:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def shot_noise(self, x, severity=1):
        # x : PIL image object
        c = [60, 25, 12, 5, 3][severity - 1]
        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / float(c), 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.shot_noise(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class ImpulseNoise:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def impulse_noise(self, x, severity=1):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.impulse_noise(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class SpeckleNoise:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def speckle_noise(self, x, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.speckle_noise(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class GlassBlur:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def glass_blur(self, x, severity=1):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(224 - c[1], c[1], -1):
                for w in range(224 - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=True), 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.glass_blur(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class DeFocusBlur:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def defocus_blur(self, x, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = list()
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))

        return np.clip(channels, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.defocus_blur(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class MotionBlur:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def motion_blur(self, x, severity=1):
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        output = BytesIO()
        # x.save(output, format='PNG')
        x.save(output, format='JPEG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

        if x.shape != (224, 224):
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = self.motion_blur(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class ZoomBlur:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def zoom_blur(self, x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += cv2.resize(clipped_zoom(x, zoom_factor), (x.shape[1], x.shape[0]))

        x = (x + out) / (len(c) + 1)
        return np.clip(x, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.zoom_blur(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Frost:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def frost(self, x, severity=1):
        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][severity - 1]
        idx = np.random.randint(5)
        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpeg'),
                    resource_filename(__name__, 'frost/frost5.jpeg'),
                    resource_filename(__name__, 'frost/frost6.jpeg')][idx]
        h, w, _ = np.array(x).shape
        frost = np.array(Image.open(filename).convert('RGB').resize((w, h)))
        return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = self.frost(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img
 


class Snow:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def snow(self, x, severity=1):
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        x = np.array(x, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        # snow_layer.save(output, format='PNG')
        snow_layer.save(output, format='JPEG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.

        l = max(x.shape[0], x.shape[1])
        snow_layer = cv2.resize(snow_layer, (l, l))
        snow_layer = snow_layer[..., np.newaxis]

        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
        a, b = x.shape[0], x.shape[1]
        return np.clip(x + snow_layer[:a, :b] + np.rot90(snow_layer, k=2)[:a, :b], 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.snow(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Spatter:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def spatter(self, x, severity=1):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return np.clip(x + color, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.spatter(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Contrast:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def contrast(self, x, severity=1):
        c = [0.4, .3, .2, .1, .05][severity - 1]

        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.contrast(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Brightness:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def brightness(self, x, severity=1):
        c = [.1, .2, .3, .4, .5][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.brightness(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Saturate:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def saturate(self, x, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.saturate(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


class Compress:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def jpeg_compression(self, x, severity=1):
        c = [25, 18, 15, 10, 7][severity - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output)

        return x

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_pil = self.jpeg_compression(img, severity)
            return fog_pil.convert('RGB')
        return img


class Pixelate:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def pixelate(self, x, severity=1):
        # x : Pil image object
        h, w = x.size
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        x = x.resize((int(h * c), int(w * c)), Image.BOX)
        x = x.resize((h, w), Image.BOX)

        return x

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_pil = self.pixelate(img, severity)
            return fog_pil.convert('RGB')
        return img


class Elastic:
    def __init__(self, p=.3):
        # p : probability of data augmentation
        self.p = p

    def elastic_transform(self, image, severity=1):
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

        image = np.array(image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1)

    def __call__(self, img):
        # img : PIL image object
        if random.uniform(0, 1) < self.p:
            severity = np.random.choice(list(range(5)), size=1)[0]
            fog_numpy = 255 * self.elastic_transform(img, severity)
            return Image.fromarray(fog_numpy.astype(np.uint8)).convert('RGB')
        return img


if __name__ == '__main__':
    img_name = 'data_aug_test/test.JPEG'
    img = Image.open(img_name)
    # img = PepperSaltNoise(p=1)(img)
    # img = ColorPointNoise(p=1)(img)
    # img = GaussianNoise(p=1)(img)
    # img = Mosaic(p=1)(img)
    # img = RGBShuffle(p=1)(img)
    # img = Rotate(p=1, angle=10)(img)
    # img = HFlip(p=1)(img)
    # img = VFlip(p=1)(img)
    # img = RandomCut(p=1)(img)
    # img = GaussianBlur(p=1)(img)
    # img = Blur(p=1)(img)
    # img = Rain(p=1)(img)
    # img = Extend(p=1)(img)
    # img = BlockShuffle(p=1)(img)
    # img = LocalShuffle(p=1)(img)
    # img = RandomPadding(p=1)(img)
    # img = Fog(p=1)(img)
    # img = ShotNoise(p=1)(img)
    # img = ImpulseNoise(p=1)(img)
    # img = SpeckleNoise(p=1)(img)
    # img = GlassBlur(p=1)(img)
    # img = DeFocusBlur(p=1)(img)
    # img = MotionBlur(p=1)(img)
    # img = ZoomBlur(p=1)(img)
    img = Frost(p=1)(img)
    # img = Snow(p=1)(img)
    # img = Spatter(p=1)(img)
    # img = Contrast(p=1)(img)
    # img = Brightness(p=1)(img)
    # img = Saturate(p=1)(img)
    # img = Compress(p=1)(img)
    # img = Pixelate(p=1)(img)
    # img = Elastic(p=1)(img)
    plt.imsave('data_aug_test/Frost.png', np.array(img))




