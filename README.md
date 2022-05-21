# RobustX: 2nd Solution for 2022 CVPR Classification Task Defense

[Homepage of Classification Task Defense](https://aisafety.sensetime.com/#/competitionDetail?id=3)

<img src="[https://github.com/ForeverPs/OPPO_6G_Data_Generation/blob/main/data/rank3.png](https://github.com/ForeverPs/Robust-Classification/blob/main/data/cvpr.png)" width="200px"/>

---

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
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Dual Attention Network for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf) (CVPR, 2019)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf) (ICCV, 2019)
