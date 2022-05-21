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
- Backbone does matter, ConvNext is better than SeResNet clearly.
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
- ConvNext(tiny) + FC + FGSM regularization + GCM(Gradient Concealment Module)
- ConvNext(tiny) + ML Decoder + FGSM regularization + GCM(Gradient Concealment Module)

<img src="https://github.com/ForeverPs/Robust-Classification/blob/main/data/gcm.png" width="700px"/>

## Pretrained Models
- Training from scratch in a two-stage manner, we provide our checkpoints.

## Training
- run `train.py`
- using `TRADES` scheme, weight of adversarial regularization equals to 1.


## Image Pre-processing
- `transforms.Resize(256)`
- `transforms.RandomResizedCrop(224)`

## Reference
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)(ICLR, 2018)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
- [A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) (CVPR, 2022)
