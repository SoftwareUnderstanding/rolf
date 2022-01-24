# Mish-Semantic Segmentation on MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset.

In 2019, Google proposed a new activation function-Mish for the machine learning. The discovery of this function makes it possible to replace the commonly used ReLU function in semantic segmentation tasks. We use MIT's ADE20k data set for experiments, and write the semantic segmentation network models according to MIT's GitHub project https://github.com/csailvision/semantic-segmentation-python. At the same time we mainly modify the basement of the semantic segmentation network models such as the ResNet, MobilNet and HRNet because these three networks was generally been used in the machine learning of the semantic segmentation. 

For example, the ResNet generally has been constituted by the Convolution layers, Pooling layers, and Softmax layer. Before the Mish function has been raised, the cell of the Convolution layer is made up of the Convolution and the Activation function, which is usually used ReLU function. Now, we change the Convolution layers that use the Mish function to replace the ReLU function. Then we use the modified network structure in the task of the semantic segmentation.

The Mish function is created for the task of the image classification. In this task it got the Top-1 Accuracy while for Generative Networks and Image Segmentation the Loss Metric. Therefore, we considered that if we use this function in the Semantic Segmentation task, we will get such a good result.
### Performance

IMPORTANT: The base ResNet in our repository is a customized (different from the one in torchvision). The base models will be automatically downloaded when needed.
---
Architecture | MultiScale Testing| Mean IoU | Pixel Accuracy(%) | Overall Score |
|:----------------------------:|:----------------:|:-----:|:----------:|:----:|
[ResNet18dilated + PPM_deepsup] |Yes|39.14|79.18|59.16|
[UperNet50]                     |Yes|41.08|79.21|60.15|
----
The training is benchmarked on a server with 4 NVIDIA Tesla V100 GPUs (16GB GPU memory). The inference speed is benchmarked a single NVIDIA Pascal 1080ti GPU, with visualization.

![image-20200314065230515](https://github.com/kukby/Mish-semantic-segmentation-pytorch/blob/master/1.png)
![image-20200314065245560](https://github.com/kukby/Mish-semantic-segmentation-pytorch/blob/master/2.png)
----
[From left to right: Test Image, Ground Truth, Predicted Result（PSPNet-ResNet18） Predicted Result（Mish-PSPNet-ResNet18）]


### Mish: A Self Regularized Non-Monotonic Neural Activation Function 

Mish is a Self Regularized Non-Monotonic Neural Activation Function. Activation Function serves a core functionality in the training process of a Neural Network Architecture and is represented by the basic mathematical representation:.

For the task of semantic segmentation. we usually found that many activation functions in the machine learning have been constructed with the  most popular amongst them being ReLU(Rectified Linear Unit;f(x)=max(0,x))

We use the Mish function  to replace the ReLU function which is used in this generally semantic segmentation.

### Dynamic scales of input for training with multiple GPUs 

For the task of semantic segmentation, it is good to keep aspect ratio of images during training. So we re-implement the `DataParallel` module, and make it support distributing data to multiple GPUs in python dict, so that each gpu can process images of different sizes. At the same time, the dataloader also operates differently. 

<sup>*Now the batch size of a dataloader always equals to the number of GPUs*, each element will be sent to a GPU. It is also compatible with multi-processing. Note that the file index for the multi-processing dataloader is stored on the master process, which is in contradict to our goal that each worker maintains its own file list. So we use a trick that although the master process still gives dataloader an index for `__getitem__` function, we just ignore such request and send a random batch dict. Also, *the multiple workers forked by the dataloader all have the same seed*, you will find that multiple workers will yield exactly the same data, if we use the above-mentioned trick directly. Therefore, we add one line of code which sets the defaut seed for `numpy.random` before activating multiple worker in dataloader.</sup>

### State-of-the-Art models

- **PSPNet** is scene parsing network that aggregates global representation with Pyramid Pooling Module (PPM). It is the winner model of ILSVRC'16 MIT Scene Parsing Challenge. Please refer to [https://arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105) for details.
- **UPerNet** is a model based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM). It doesn't need dilated convolution, an operator that is time-and-memory consuming. *Without bells and whistles*, it is comparable or even better compared with PSPNet, while requiring much shorter training time and less GPU memory. Please refer to [https://arxiv.org/abs/1807.10221](https://arxiv.org/abs/1807.10221) for details.
- **HRNet** is a recently proposed model that retains high resolution representations throughout the model, without the traditional bottleneck design. It achieves the SOTA performance on a series of pixel labeling tasks. Please refer to [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514) for details.


## Supported models



We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. We have provided some pre-configured models in the ```config``` folder.

Encoder:
=======

We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. We have provided some pre-configured models in the ```config``` folder.

Encoder:

- MobileNetV2dilated
- ResNet18/ResNet18dilated
- ResNet50/ResNet50dilated
- ResNet101/ResNet101dilated
- HRNetV2 (W48)

Decoder:

- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)
- UPerNet (Pyramid Pooling + FPN head, see [UperNet](https://arxiv.org/abs/1807.10221) for details.)

## Environment

The code is developed under the following configurations.

- Hardware: >=4 GPUs for training, >=1 GPU for testing (set ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 18.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=1.0.0***
- Dependencies: numpy, scipy, opencv, yacs, tqdm

## Quick start: Test on an image using our trained model 

1. Here is a simple demo to do inference on a single image:

```bash
chmod +x demo_test.sh
./demo_test.sh
```

This script downloads a trained model (ResNet50dilated + PPM_deepsup) and a test image, runs the test script, and saves predicted segmentation (.png) to the working directory.

2. To test on an image or a folder of images (```$PATH_IMG```), you can simply do the following:

```
python3 -u test.py --imgs $PATH_IMG --gpu $GPU --cfg $CFG
```

## Training

1. Download the ADE20K scene parsing dataset:

```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```

2. Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, checkpoints by default are saved in folder ```ckpt```.

```bash
python3 train.py --gpus $GPUS --cfg $CFG 
```

- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

​       For example, you can start with our provided configurations: 

* Train MobileNetV2dilated + C1_deepsup

```bash
python3 train.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Train ResNet50dilated + PPM_deepsup

```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Train UPerNet101

```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```

3. You can also override options in commandline, for example  ```python3 train.py TRAIN.num_epoch 10 ```.


## Evaluation

1. Evaluate a trained model on the validation set. Add ```VAL.visualize True``` in argument to output visualizations as shown in teaser.

   For example:

* Evaluate MobileNetV2dilated + C1_deepsup

```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Evaluate ResNet50dilated + PPM_deepsup

```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Evaluate UPerNet101

```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```

## Reference

If you find the code or pre-trained models useful, please cite the following papers:

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }

Misra D. Mish: A Self Regularized Non-Monotonic Neural Activation Function[J]. arXiv preprint arXiv:1908.08681, 2019.(https://github.com/digantamisra98/Mish)

```
@misc{misra2019mish,
    title={Mish: A Self Regularized Non-Monotonic Neural Activation Function},
    author={Diganta Misra},
    year={2019},
    eprint={1908.08681},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

### TO DO

test all the Semantic-Segmentation Neural Network Model by Mish.

Complete all test data by Mish in the task of Semantic-segmentation.
