# UNet: Cell segmentation with PyTorch
![alt text](/image/Screenshot%20from%202020-02-04%2017-42-19.png)
![alt text](/image/Screenshot%20from%202020-02-04%2017-42-57.png)
![alt text](/image/Screenshot%20from%202020-02-04%2017-43-16.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch.

This model was trained from scratch with 150 images (with data augmentation)This score could be improved with more training, data augmentation, fine tuning, playing with CRF post-processing, and applying more weights on the edges of the masks.



## Usage
**Note : Use Python 3**

The input images and target masks should be in the `data/cells/scans` and `data/cells/labels` folders respectively.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
