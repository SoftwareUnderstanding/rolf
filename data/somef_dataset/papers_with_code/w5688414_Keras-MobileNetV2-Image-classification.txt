# MobileNet v2 
A Python 3 and Keras 2 implementation of MobileNet V2 and provide train method.  

According to the paper: [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381).


## Requirement
- OpenCV 3.4
- Python 3.5    
- Tensorflow-gpu 1.5.0  
- Keras 2.2


## MobileNet v2 and inverted residual block architectures

**MobileNet v2:**  

Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 X 3 kernels. The expansion factor t is always applied to the input size.

![MobileNetV2](/images/net.jpg)

**Bottleneck Architectures:**

![residual block architectures](/images/stru.jpg)


## Train the model

The recommended size of the image in the paper is 224 * 224. The ```data\convert.py``` file provide a demo of resize cifar-100 dataset to this size.

change the image root directory in the train.py

you need to prepare train.txt and valid.txt
the file format is like:
```
image,label
scence#3/occlusion/50%/glasses_04/frame0215.jpg,glasses_04
scence#1/pixel_color/30/pencil_01/frame0093.jpg,pencil_01
scence#2/illumination/Normal/ladle_01/frame0214.jpg,ladle_01
scence#1/Illumination/Strong/scissors_02/frame0191.jpg,scissors_02
scence#3/clutter/High/toy_05/frame0110.jpg,toy_05
scence#2/pixel/30-200/ladle_04/frame0214.jpg,ladle_04
scence#3/pixel/200/toy_03/frame0221.jpg,toy_03
scence#1/Illumination/Normal/stapler_03/frame0153.jpg,stapler_03
scence#1/pixel_color/200/pencil_05/frame0101.jpg,pencil_05
scence#1/pixel_color/30-200/paper_cutter_03/frame0051.jpg,paper_cutter_03
scence#1/clutter_color/High/stapler_01/frame0173.jpg,stapler_01
scence#1/clutter_color/High/paper_cutter_01/frame0089.jpg,paper_cutter_01
scence#3/occlusion/50%/glasses_04/frame0248.jpg,glasses_04
scence#2/pixel/30-200/cup_01/frame0096.jpg,cup_01
scence#2/pixel/200/cup_01/frame0051.jpg,cup_01
```

**Run command below to train the model:**

```
python3 train.py --classes num_classes --batch batch_size --epochs epochs --size image_size  --train train.txt --valid valid.txt
```

The ```.h5``` weight file was saved at model folder. If you want to do fine tune the trained model, you can run the following command. However, it should be noted that the size of the input image should be consistent with the original model.

```
python3 train.py --classes num_classes --batch batch_size --epochs epochs --size image_size --weights weights_path --tclasses pre_classes  --train train.txt --valid valid.txt
```

**Parameter explanation**

- --classes, The number of classes of dataset.  
- --size,    The image size of train sample.  
- --batch,   The number of train samples per batch.  
- --epochs,  The number of train iterations.  
- --weights, Fine tune with other weights.  
- --tclasses, The number of classes of pre-trained model.
- --train,   train file name
- --valid,  valid file name

## Reference

	@article{MobileNetv2,  
	  title={Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentatio},  
	  author={Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen},
	  journal={arXiv preprint arXiv:1801.04381},
	  year={2018}
	}


[MobileNetV2](https://github.com/xiaochus/MobileNetV2)


## Copyright
See [LICENSE](LICENSE) for details.


