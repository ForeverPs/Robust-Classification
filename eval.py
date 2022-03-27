import numpy as np


def get_acc(predict, target):
    predict = predict.detach().cpu().squeeze().numpy()
    target = target.detach().cpu().squeeze().numpy()
    acc = np.sum(predict == target) / len(predict)
    return acc