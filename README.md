# Gradient Concealment: Free Lunch for Defending Adversarial Attacks

[![Rank 2](https://img.shields.io/badge/RobustX-2nd%20Solution%20of%20CVPR%202022%20Robust%20Classification%20Challenge-brightgreen.svg?style=flat-square)](https://github.com/ForeverPs/Robust-Classification)

<img src="https://github.com/ForeverPs/Robust-Classification/blob/main/data/cvpr.png" width="800px"/>

---

[Paper](https://arxiv.org) |
[Checkpoints](MODEL_ZOO.md)  |
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

## AR Results on ImageNet

| Backbone | Method | Top 1 Acc | FGSM Linf=8/255 | PGD L1=1600 | PGD L2=8.0 | PGD Linf=8/255 | C&W L2=8.0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | Vanilla | 77.89 | 31.77 | 46.0 | 41.6 | 48M | 267G |
| ResNet50 | GCM | 78.57 | 95.18 | 48.5 | 43.3 | 69M | 359G |
| Swin-T | Cascade Mask R-CNN | ImageNet-1K | 3x | 50.4 | 43.7 | 86M | 745G |
| Swin-S | Cascade Mask R-CNN | ImageNet-1K |  3x | 51.9 | 45.0 | 107M | 838G |
| Swin-B | Cascade Mask R-CNN | ImageNet-1K |  3x | 51.9 | 45.0 | 145M | 982G |
| Swin-T | RepPoints V2 | ImageNet-1K | 3x | 50.0 | - | 45M | 283G |
| Swin-T | Mask RepPoints V2 | ImageNet-1K | 3x | 50.3 | 43.6 | 47M | 292G |
| Swin-B | HTC++ | ImageNet-22K | 6x | 56.4 | 49.1 | 160M | 1043G |
| Swin-L | HTC++ | ImageNet-22K | 3x | 57.1 | 49.5 | 284M | 1470G |
| Swin-L | HTC++<sup>*</sup> | ImageNet-22K | 3x | 58.0 | 50.4 | 284M | - |

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)(ICLR, 2018)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
- [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) (CVPR, 2022)
