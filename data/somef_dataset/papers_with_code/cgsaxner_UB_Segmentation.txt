# Automatic urinary bladder segmentation in CT images using deep learning
A framework for urinary bladder segmentation in CT images using deep learning.

Contains code to train and test two different deep neural network architectures for semantic segmentation using training and testing data obtained from combined PET/CT scans. 

## Requirements
To use the framework, you need:

1. [Python](https://www.python.org/download/releases/3.5/) 3.5 with the packages specified in the [requirements.txt](https://github.com/cgsaxner/UB_Segmentation/blob/master/requirements.txt) file
2. [TensorFlow](https://www.tensorflow.org/versions/r1.3/) 1.3
3. [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) library

## Data
Our networks were trained and tested on the publically available [RIDER Lung PET CT Dataset](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+PET-CT). 


The data was preprocessed and prepared using the MeVisLab network [Exploit 18F-FDG enhanced urinary bladder in PET data for Deep Learning Ground Truth Generation in CT scans](https://github.com/cgsaxner/DataPrep_UBsegmentation).


This software produces ground-truth segmentations of the urinary bladder in CT using the co-registered PET data. PET radiotracer 18F-FDG accumulates in the urinary bladder, therefore, this organ can be distinguished using simple thresholding. Furthermore, data augmentation is applied using MeVisLab software. For further information, please refer to the corresponding paper:

[Exploit 18F-FDG Enhanced Urinary Bladder in PET Data for Deep Learning Ground Truth Generation in CT Scans.](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10578/105781Z/Exploit-sup18-supF-FDG-enhanced-urinary-bladder-in-PET-data/10.1117/12.2292706.short?SSO=1)


## Functionalities

- **Creating TFRecords files for training and testing data.** 
The script [`make_tfrecords_dataset.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/make_tfrecords_dataset.py) contains code to convert a directory of image files to the TensorFlow recommended file format TFRecords. TFRecords files are easy and fast to process in TensorFlow.

- **Training networks.**
The scripts [`FCN_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_training.py) and [`ResNet_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_training.py) contain code for training two different neural network architectures for semantic segmentation.
FCN is based on [FCN-8s by Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) using [pre-trained VGG](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py).
ResNet is based on [DeepLab by Chen et al.](https://arxiv.org/pdf/1606.00915.pdf]) using [pre-trained ResNet V2](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py).

- **Testing networks.**
The scripts [`FCN_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_testing.py) and [`ResNet_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_testing.py) contain code for testing the previously trained networks.

- **Evaluation metrics.**
The file [`metrics.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/metrics.py) contains functions to calculate following metrics for evaluating segmentation results:
  - True Positive Rate (TPR)
  - True Negative Rate (TNR)
  - Intersection over union (Jaccard Index, IoU)
  - Dice-Sorensen coefficient (DSC)
  - Hausdorff distance (HD)
  
## Getting started

To use the framework for creating a tf-records file:
1. Place your images and ground truth in folders called **Images** and **Labels**, respectively.
2. Specify the path to your data, the desired filename and the desired image size in [`make_tfrecords_dataset.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/make_tfrecords_dataset.py).
3. Run the script!

To use the framework for training:
1. Download the pre-trained model checkpoint you want to use from [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) and place it in a **\Checkpoints** folder in your project repository.
2. Specify your paths in the top section of [`FCN_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_training.py) or [`ResNet_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_training.py).
3. Run the script!

To use the framework for testing:
1. Specify your paths in the top section of [`FCN_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_testing.py) or [`ResNet_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_testing.py).
2. Run the script!
  
## Acknowledgements
Parts of the code are based on [tf-image-segmentation](https://github.com/warmspringwinds/tf-image-segmentation). If using, please cite his paper:

      @article{pakhomov2017deep,
        title={Deep Residual Learning for Instrument Segmentation in Robotic Surgery},
        author={Pakhomov, Daniil and Premachandran, Vittal and Allan, Max and Azizian, Mahdi and Navab, Nassir},
        journal={arXiv preprint arXiv:1703.08580},
        year={2017}
      }
    
## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/cgsaxner/UB_Segmentation/blob/master/LICENSE) file for details.

## Using the framework

If you use the framework, please cite the following paper:

Gsaxner, Christina et al. **Exploit 18F-FDG Enhanced Urinary Bladder in PET Data for Deep Learning Ground Truth Generation in CT Scans.** SPIE Medical Imaging 2018.

    @inproceedings{gsaxner2018exploit,
      title={Exploit 18 F-FDG enhanced urinary bladder in PET data for deep learning ground truth generation in CT scans},
      author={Gsaxner, Christina and Pfarrkirchner, Birgit and Lindner, Lydia and Jakse, Norbert and Wallner, J{\"u}rgen and Schmalstieg, Dieter and Egger, Jan},
      booktitle={Medical Imaging 2018: Biomedical Applications in Molecular, Structural, and Functional Imaging},
      volume={10578},
      pages={105781Z},
      year={2018},
      organization={International Society for Optics and Photonics}
    }



