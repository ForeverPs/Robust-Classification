import cv2
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


def extend(src, offsets=5):
    dst = src.copy()
    rows, cols, _ = src.shape
    x = np.array(list(range(rows)))
    y = np.array(list(range(cols)))
    yv, xv = np.meshgrid(y, x)

    random1 = np.random.randint(0, offsets, size=yv.shape)
    random2 = np.random.randint(0, offsets, size=xv.shape)

    yv_random = yv + random1
    xv_random = xv + random2

    xv_random[xv_random >= rows] = rows - 1
    yv_random[yv_random >= cols] = cols - 1
    dst[xv, yv] = src[xv_random, yv_random]
    return dst


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(x, y):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(x.size()[0])
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def rain_noise(img, value, angle, beta):
    noise = get_noise(img, value=value)
    rain = rain_blur(noise, length=10, angle=angle, w=3)
    result_img = alpha_rain(rain, img, beta=beta)
    return result_img


def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def alpha_rain(rain, img, beta=0.8):
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result


def motion_blur(image, degree=12, angle=45):
    # image : numpy array
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def draw_hist(img_dict):
    x = [str(i) for i in range(20)]
    y = [img_dict[name] for name in x]
    plt.bar(x, y, align='center', color='b', tick_label=x, alpha=0.6)

    plt.xlabel('image category')
    plt.ylabel('image counts')
    plt.title('Distribution within Training Images')

    plt.grid(True, axis='y', ls=':', color='r', alpha=0.3)
    plt.show()


def parse_txt(txt_name):
    with open(txt_name, 'r') as f:
        lines = f.readlines()
    f.close()

    training_pairs = list()
    img_counts = dict()
    for line in tqdm.tqdm(lines):
        img_name, label = line.strip().split(' ')
        training_pairs.append((img_name, int(label)))
        if label not in img_counts.keys():
            img_counts[label] = 0
        img_counts[label] += 1
    return training_pairs, img_counts


if __name__ == '__main__':
    txt_name = 'data/train_phase1/label.txt'
    training_pairs, img_counts = parse_txt(txt_name)
    draw_hist(img_counts)