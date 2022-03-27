import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
    txt_name = 'train_phase1/label.txt'
    training_pairs, img_counts = parse_txt(txt_name)
    draw_hist(img_counts)