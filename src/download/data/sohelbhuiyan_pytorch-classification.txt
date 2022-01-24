# ImageNet Classification with Weight Excitation (MWE / LWE)
This project is our implementation of Weight Standardization for ImageNet classification with ResNet and ResNeXt. The project is forked from pytorch-classification.

It is an unofficial implementation of the "Weight Excitation: Built-in Attention Mechanisms in Convolutional Neural Networks" paper. The link of the paper is as follows:
https://link.springer.com/chapter/10.1007%2F978-3-030-58577-8_6

If you find this project helpful, please consider citing our paper.

```
@inproceedings{quader2020weight,
  title={Weight Excitation: Built-in Attention Mechanisms in Convolutional Neural Networks},
  author={Quader, Niamul and Bhuiyan, Md Mafijul Islam and Lu, Juwei and Dai, Peng and Li, Wei},
  booktitle={European Conference on Computer Vision},
  pages={87--103},
  year={2020},
  organization={Springer}
}
```

## Training
Please see the [Training recipes](TRAINING.md) for how to train the models.
NOTE: In reality we do not use batch size 1 per GPU for training since it is so slow. Because GN+WS does not use any batch knowledge, setting batch size to 256 and iteration size to 1 (for example) is equivalent to setting batch size to 1 and iteration size to 256. Therefore, to speed up training, we set batch size to large values and use the idea of iteration size to simulate micro-batch training. We provide the following training scripts to get the reported results. 4 GPUs with 12GB each are assumed.

ResNet-50:
```
python -W ignore imagenet.py -a l_resnet50 --data ~/dataset/ILSVRC2012/ --epochs 90 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/resnet50 --gpu-id 0,1,2,3
```
ResNet-101:
```
python -W ignore imagenet.py -a l_resnet101 --data ~/dataset/ILSVRC2012/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/resnet101 --gpu-id 0,1,2,3 --train-batch 128 --test-batch 128
```
ResNeXt-50 32x4d:
```
python -W ignore imagenet.py -a l_resnext50 --base-width 4 --cardinality 32 --data ~/dataset/ILSVRC2012/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/resnext50-32x4d --gpu-id 0,1,2,3 --train-batch 128 --test-batch 128
```

ResNeXt-101 32x4d:
```
python -W ignore imagenet.py -a l_resnext101 --base-width 4 --cardinality 32 --data ~/dataset/ILSVRC2012/ 
```

# Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).

