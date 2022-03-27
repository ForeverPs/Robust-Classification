# RobustX

Robust classification in the presence of polluted images 

---

## Datasets
- train_phase1/images/ : 22987 images for training
- train_phase1/label.txt : ground-truth file
- track1_test1/ : 20000 images for testing

## Data Augmentation Scheme
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

## Architectures
- ResNet50 (maybe SeResNet) + Dual Attention + FGSM regularization + (Energy Ranking / Prototype Cluster)

## Pretrained Models
- Pretrained models on ImageNet are forbidden to use.

## Training
- run `train.py`

## Validation
- Done. 1k images in testing set are labeled for validation

## Boost Scheme
- Somehow forbidden to use.

## Reference
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Dual Attention Network for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf) (CVPR, 2019)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
