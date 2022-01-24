# TF2_U2Model
- This is the U2 model for segmentation implemented in Tensorflow 2.3

- With training data set consisting of around 3000 dicom hand images, we can achive test IOU of 0.926 in 300 epoch for palm segmentation.

- We' ve modified the loss function.



## Reference from 
- U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection

  https://arxiv.org/pdf/2005.09007.pdf
- pytorch implemtation (We revised the code to tensorflow version)

  https://github.com/NathanUA/U-2-Net





![pterygium](https://github.com/soxHenry433/TF2_U2Model/blob/master/Test/1D0734605CB1FF86A792C14BB6A794616FA37246-HR-20181122_0.png "Predicted images")


|            |           |
|:---------:|:---------:|
|**Raw Image**|**Predicted**| 
|**Raw Mask**|**Predicted**|


