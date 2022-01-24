# CNN-segmentation-for-Lung-cancer-OARs
a deep convolutional neural network (CNN)-based automatic segmentation technique was applied to the multiple organs at risk (OARs) in CT images of lung cancer

The Uploads of the our article "Preliminary comparison of the automatic segmentation of multiple organs at risk in CT images of lung cancer between deep convolutional neural network-based and atlas-based techniques".
The uploads include  the neural network architecture and the detail architecture diagrams.

This is a generic 3D volume U-Net convolutional network implementation as proposed by `Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`

The loss function is dice similarity coefficient (DSC) with variable weight.

### Requirements
- Python 3.5+
  - tensorflow-gpu 1.3+
  - keras 2.1+
  - numpy 1.12.0+
