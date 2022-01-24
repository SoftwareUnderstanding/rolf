# Cartoon-style-Stickers-Generator

## Background
Stikers are very common to express your emotion in a communication but the cost to generate it is not cheap. However, using AI to generate cartoon-style stickers is one of the solution. I combined instance segmentation and style transfer in this project for generating stickers on your own images.

## Example
![](https://i.imgur.com/a8mBztA.jpg)


## Requirements
- torch==1.7.1
- torchvision==0.8.2
- cv2==4.5.3

## Experiment
Load the pretrained model trained_netG.pth from https://drive.google.com/file/d/1aBlInboCznUhjULZfk4i7ENnLdcRW0jA/view?usp=sharing
```
cp trained_netG.pth Cartoon-style-Stickers-Generator/checkpoints/trained_netG.pth
```
Apply on your own image which follows jpg format
```
python3 main.py --image_path 'demo.jpg'
```

## References
- https://arxiv.org/abs/1703.06870
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf
- https://github.com/FilipAndersson245/cartoon-gan
