# RobustX

Robust classification in the presence of polluted images 

---

## Datasets
- train_phase1/images/ : 22987 images for training
- train_phase1/lanel.txt : ground-truth file
- track1_test1/ : 20000 images for testing

## Data Augmentation Scheme
- `Currently Support`
- PepperSaltNoise
- ColorPointNoise
- GaussianNoise
- Mosaic in black/gray/white/color
- RGBShuffle
- Rotate
- HorizontalFlip
- VerticalFlip
- RandomCut
- MotionBlur
- GaussianBlur
- ConventionalBlur

## Architectures
- ResNet50 (maybe SeResNet) + Dual Attention + FGSM regularization

## Pretrained Models
- ImageNet Pretrained

## Training
- To do

## Validation
- Need to label about 1k images in testing set used for validation

## Boost Scheme
- To do

## Reference
- [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR, 2016)
- [Dual Attention Network for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf) (CVPR, 2019)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (CVPR, 2018)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (NeurIPS, 2014)
