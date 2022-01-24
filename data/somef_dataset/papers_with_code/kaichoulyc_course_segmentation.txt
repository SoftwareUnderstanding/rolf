# Comparision of few semantic segmentation algorithms

For comparison I took 2 popular algorithms: UNet[1] (with various encoders) and DeepLab V3+[2]

## Train and validation loss measurements

![ResNet34](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/resnet34_loss.png)

![ResNext50](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/resnext50_loss.png)

![SeNet150](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/senet150_loss.png)

![DeepLab V3+](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/deeplab_loss.png)

![Train loss comparision](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/train_comp.png)

![Validation loss comparision](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/valid_comp.png)

![Validation IoU comparision](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/valid_iou_comp.png)


## Final scores

![During train](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/Tab_1.png)

![Test set scores](https://github.com/kaichoulyc/course_segmentation/blob/master/figures/Tab_2.png)


## References

[1] https://arxiv.org/abs/1505.04597
[2] https://arxiv.org/abs/1802.02611
