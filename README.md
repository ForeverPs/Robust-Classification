# Gradient Concealment: Free Lunch for Defending Adversarial Attacks

[![Rank 2](https://img.shields.io/badge/RobustX-2nd%20Solution%20of%20CVPR%202022%20Robust%20Classification%20Challenge-brightgreen.svg?style=flat-square)](https://github.com/ForeverPs/Robust-Classification)

<img src="https://github.com/ForeverPs/Robust-Classification/blob/main/data/cvpr.png" width="800px"/>

---

[Paper](https://arxiv.org) |
[Checkpoints]()  |
[Homepage](https://aisafety.sensetime.com/#/competitionDetail?id=3)

Official PyTorch Implementation

> Sen Pei, Jiaxi Sun, Xin Zhang, Qing Li
> <br/> Institute of Automation, Chinese Academy of Sciences

## Conclusion
- Backbone does matter, ConvNext is better than SeResNet.
- Randomization is efficient for defending adversarial attacks.
- Data augmentation is vital for improving the classification performance, reducing overfitting.
- Gradient concealment dramatically improves AR metric of classifiers in presence of perturbed images.


## Datasets
- train_phase1/images/ : 22987 images for training
- train_phase1/label.txt : ground-truth file
- track1_test1/ : 20000 images for testing
- train_p2 : 127390 images within 100 categories

## Data Augmentation Schemes
`data_aug.py supports the following operations currently:`
- PepperSaltNoise
- ColorPointNoise
- GaussianNoise
- Mosaic in black / gray / white / color
- RGBShuffle / ColorJitter
- Rotate
- HorizontalFlip / VerticalFlip
- RandomCut
- MotionBlur / GaussianBlur / ConventionalBlur
- Rain / Snow
- Extend
- BlockShuffle
- LocalShuffle (for learning local spatial feature)
- RandomPadding (for defense of adversarial attacks)

![avatar](https://github.com/ForeverPs/Robust-Classification/blob/main/data_aug_test/demo.png)

## Adversarial Defense Schemes
- Adversarial training using fast gradient sign method.
- Resize and pad the input images for mitigating adversarial effects.
- Gradient concealment module for hiding the vulnerable direction of classifier's gradient.

## Architectures
- ConvNext(tiny) + FC + FGSM regularization + GCM(Gradient Concealment Module) + Randomization
- ConvNext(tiny) + ML Decoder + FGSM regularization + GCM(Gradient Concealment Module) + Randomization

<img src="https://github.com/ForeverPs/Robust-Classification/blob/main/data/gcm.png" width="700px"/>

## Pretrained Models
- Training from scratch in a two-stage manner, we provide our checkpoints.

## Training
- run `train.py`
- using `TRADES` scheme, weight of adversarial regularization equals to 1.


## Image Pre-Processing
- `transforms.Resize(256)`
- `transforms.RandomResizedCrop(224)`

## Some AR Results on ImageNet

| Model | Method | Top 1 Acc | FGSM Linf=8/255 | PGD L1=1600 | PGD L2=8.0 | PGD Linf=8/255 | C&W L2=8.0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 | Vanilla | 77.89 | 31.77 | 0.01 | 0.01 | 0.00 | 0.21 |
| ResNet-50 | GCM | 78.57 | 95.18 | 94.82 | 94.41 | 97.38 | 95.11 |
| WideResNet-50 | Vanilla | 78.21 | 20.88 | 0.36 | 0.61 | 0.50 | 0.21 |
| WideResNet-50 | GCM | 78.08 |  96.06 | 94.46 | 94.51 | 97.69 | 95.66 |
| DenseNet-121 | Vanilla | 74.86 |  16.82 | 0.04 | 0.05 | 0.06 | 0.12 |
| DenseNet-121 | GCM | 74.71 | 94.98 | 94.31 | 94.08 | 97.16 | 95.49 |
| EfficientNet-B4 | Vanilla | 71.52 | 1.23 | 0.36 | 0.28 | 0.20 | 1.88 |
| EfficientNet-B4 | GCM | 71.76 | 94.68 | 89.95 | 90.87 | 97.97 | 93.07 |
| ViT-B/16 | Vanilla | 79.46 | 15.86 | 0.00 | 0.00 | 0.00 | 0.90 |
| ViT-B/16 | GCM | 79.47 | 92.24 | 94.94 | 95.07 | 98.24 | 93.31 |
| Swin-Transformer-S | Vanilla | 82.93 | 16.93 | 0.20 | 0.00 | 0.00 | 0.76 |
| Swin-Transformer-S | GCM | 82.79 | 94.38 | 90.71 | 91.04 | 98.77 | 92.31 |

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)(ICLR, 2018)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
- [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) (CVPR, 2022)
