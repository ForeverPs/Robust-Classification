# Gradient Concealment: Free Lunch for Defending Adversarial Attacks

`RobustX: 2nd Solution for 2022 CVPR Classification Task Defense`

<img src="https://github.com/ForeverPs/Robust-Classification/blob/main/data/cvpr.png" width="800px"/>

---

[Paper](https://arxiv.org/abs/2009.14119) |
[Checkpoints](MODEL_ZOO.md)  |
[Homepage](https://aisafety.sensetime.com/#/competitionDetail?id=3)

Official PyTorch Implementation

> Sen Pei, Jiaxi Sun, Xin Zhang, Qing Li
> <br/> Institute of Automation, Chinese Academy of Sciences

## Conclusion
- training more epochs = better performance, verified on SeResNet18, SeResNet34 and SeResNet50
- Pretrained models are helpful for improving robustness (Fordidden to use)
- Add more mlp layers, using `LeakyReLU(0.1)` and `Dropout(0.3)`



## Datasets
- train_phase1/images/ : 22987 images for training
- train_phase1/label.txt : ground-truth file
- track1_test1/ : 20000 images for testing

## Data Augmentation Scheme（up to 31 kinds）
`data_aug.py supports the following operations currently:`
- PepperSaltNoise
- ColorPointNoise
- GaussianNoise
- Mosaic in black / gray / white / color
- RGBShuffle
- Rotate
- HorizontalFlip
- VerticalFlip
- RandomCut
- MotionBlur
- GaussianBlur
- ConventionalBlur
- Rain
- Extend
- BlockShuffle
- LocalShuffle (for learning local spatial feature)
- RandomPadding (for defense of adversarial attacks)

![avatar](https://github.com/ForeverPs/Robust-Classification/blob/main/data_aug_test/demo.png)

## Architectures
- SeResNet18 + ML Decoder + FGSM regularization
![avatar](https://github.com/ForeverPs/Robust-Classification/blob/main/data_aug_test/senet.png)

## Pretrained Models
- Training from scratch...

## Training
- run `train.py`
- using `TRADES` scheme, weight of adversarial regularization equals to 1.

## Validation
- test images for validation.

## Boost Scheme
- `torch.FiveCrop(224)` introduces no further improvement

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
- [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) (CVPR, 2022)
