# PyTorch Implementation of CIFAR-10 Image Classification Pipeline Using VGG Like Network

We present here our solution to the famous machine learning problem of image classification with [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset with 60000 labeled images. The aim is to learn and assign a category for these `32x32` pixel images.


## Dataset

The CIFAR-10 dataset, as it is provided, consists of 5 batches of training images which sum up to 50000 and a batch of 10000 test images.

Each test batch consists of exactly 1000 randomly-selected images from each class. The training batches contain images in random order, some training batches having more images from one class than another. Together, the training batches contain exactly 5000 images from each class.

Here we have used for training and validation purposes only the 50000 images originally meant for training. [Stratified K-Folds cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) is used to split the data so that the percentage of samples for each class is preserved. Several other reported implementations use the data as it is given and use the given 10000 sample testing set straight for validation. Instead we use the 10000 sample test set for evaluating our trained model.


## Model

We have made a PyTorch implementation of [Sergey Zagoruyko](https://github.com/szagoruyko/cifar.torch) VGG like network with BatchNormalization and Dropout for the task.

```
DataParallel(
  (module): VGGBNDrop(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): Dropout(p=0.3)
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU(inplace)
      (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (10): ReLU(inplace)
      (11): Dropout(p=0.4)
      (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (14): ReLU(inplace)
      (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (16): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (18): ReLU(inplace)
      (19): Dropout(p=0.4)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace)
      (23): Dropout(p=0.4)
      (24): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (28): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (30): ReLU(inplace)
      (31): Dropout(p=0.4)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (34): ReLU(inplace)
      (35): Dropout(p=0.4)
      (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (38): ReLU(inplace)
      (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (42): ReLU(inplace)
      (43): Dropout(p=0.4)
      (44): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (45): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (46): ReLU(inplace)
      (47): Dropout(p=0.4)
      (48): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (49): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (50): ReLU(inplace)
      (51): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    )
    (classifier): Sequential(
      (0): Dropout(p=0.5)
      (1): Linear(in_features=512, out_features=512, bias=True)
      (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace)
      (4): Dropout(p=0.5)
      (5): Linear(in_features=512, out_features=10, bias=True)
    )
  )
)
```


## Data Augmentations

In this implementation we only use [horizontal flips](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.RandomFlip). We pad the images into size `34x34` using [reflective padding](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.PadTransform) and then crop the images back into size `32x32`. [Random cropping](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.CropTransform) is used as an augmentation in the training and then [center cropping](https://mipt-oulu.github.io/solt/transforms.html#solt.transforms.CropTransform) in the validation phase. Moreover, [`solt`](https://mipt-oulu.github.io/solt/) is used for the data augmentations.

In their experiments, [Sergey Zagoruyko and Nikos Komodakis](https://github.com/szagoruyko/wide-residual-networks) seem to have used whitened data. We use here the original data.

`YUV` color space was proposed to be used by [Sergey Zagoruyko](https://github.com/szagoruyko/cifar.torch). We have run our experimets without the `RGB` to `YUV` conversion.

Data is normalized in the usual way with mean and standard deviation calculated across the 50000 images, as it can, e.g., speed up the training.

## Setting up the data for training

From PyCharm Terminal

```
$ python build_dataset.py --dataset CIFAR10
```

## Training

From PyCharm Terminal

```
$ python run_training.py --dataset_name CIFAR10 --num_classes 10 --experiment vggbndrop --bs 128 --optimizer sgd --lr 0.1 --lr_drop "[160, 260]" --n_epochs 300 --wd 5e-4 --learning_rate_decay 0.2 --n_threads 12 --color_space rgb --set_nesterov True
```


## Results for CIFAR-10

Here we provide the results related to the `VGGBNDrop` model proposed by [Sergey Zagoruyko](https://github.com/szagoruyko/cifar.torch) using `SGD` as optimizer.

### Training and validation

As can be seen from the curves representing loss over time, the model starts to overfit around epoch 164.

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/Loss_fold_0_2019_02_25_06_20.png" title="Loss over time">
</p>

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/Accuracy_fold_0_2019_02_25_06_20.png" title="Validation accuracy over time">
</p>

From the confusion matrices below related to the validation accuracy curve, we can see how the learning progresses.

Epoch 40:

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/CM_fold_0_epoch_39_2019_02_24_19_54.png" title="Confusion Matrix, Validation, Epoch 40">
</p>

Epoch 80:

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/CM_fold_0_epoch_79_2019_02_24_20_51.png" title="Confusion Matrix, Validation, Epoch 80">
</p>

Epoch 120:

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/CM_fold_0_epoch_119_2019_02_24_22_02.png" title="Confusion Matrix, Validation, Epoch 120">
</p>

Epoch 160:

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/CM_fold_0_epoch_159_2019_02_24_23_27.png" title="Confusion Matrix, Validation, Epoch 160">
</p>

### Evaluation

Evaluation has been run using the model for which the validation loss was the best (see [`session`](https://github.com/aisosalo/CIFAR-10/blob/master/imageclassification/training/session.py#L152-L178) for details).

<p align="center">
  <img src="https://github.com/aisosalo/CIFAR-10/blob/master/plots/CM_evaluation_2019_02_25.png" title="Confusion Matrix, Evaluation">
</p>


## Acknowledgements

[Aleksei Tiulpin](https://github.com/lext/) is acknowledged for kindly providing access to his pipeline scripts and giving his permission to reproduce and modify his pipeline for this task.

[Research Unit of Medical Imaging, Physics and Technology](https://www.oulu.fi/mipt/) is acknowledged for making it possible to run the experiments.


## Authors

[Antti Isosalo](https://github.com/aisosalo/), [University of Oulu](https://www.oulu.fi/university/), 2018-


## References

### Model Architecture

* Zagoruyko, Sergey, and Nikos Komodakis. "[Wide Residual Networks](https://arxiv.org/abs/1605.07146)." Proceedings of the British Machine Vision Conference (BMVC), 2016.

* Zagoruyko, Sergey. "[92.45% on CIFAR-10](https://github.com/szagoruyko/cifar.torch)." 2015

### Data Augmentation

* Tiulpin, Aleksei, "[Streaming Over Lightweight Data Transformations](https://mipt-oulu.github.io/solt/)." Research Unit of Medical Imaging, Physics and Technology, University of Oulu, Finalnd, 2018.

### Dataset

* Krizhevsky, Alex, and Geoffrey Hinton. "[Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)." Vol. 1. No. 4. Technical Report, University of Toronto, 2009.

* Benenson, Rodrigo. "[Are we there yet](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)." 2016.

* Recht, Benjamin, Roelofs, Rebecca, Schmidt, Ludwig, and Shankar, Vaishaal. "[Do CIFAR-10 Classifiers Generalize to CIFAR-10?](https://arxiv.org/abs/1806.00451)." arXiv preprint arXiv:1806.00451, 2018.
