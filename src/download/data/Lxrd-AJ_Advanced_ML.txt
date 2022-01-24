# Advanced_ML
# Action plan for the DSTL Image Segmentation

# Data Preprocessing (Philipp and James)
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection 
* https://www.kaggle.com/alijs1/squeezed-this-in-successful-kernel-run/code
* General overview: https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/

- [x] Data Importing (Goal: 1x 20-channels image)
	- [x] Gray-Scale
	- [x] 3-Band
	- [x] 16-Band
- [x] Creatable Torch DataLoader
- [x] Data Transformations
	- [ ] Up-scaling	
	- [x] Random-cropping
	- [x] Resizing
	- [ ] Rotations
- [x] Mask-to-Polygon Transformation (in data_import.py "mask_to_polygons" function)

- [ ] Data visualization
	- https://www.kaggle.com/torrinos/exploration-and-plotting?scriptVersionId=558039
	- https://www.kaggle.com/visoft/export-pixel-wise-mask/code
	- https://www.kaggle.com/jeffhebert/stitch-a-16-channel-image-together
	- https://www.kaggle.com/torrinos/exploration-and-plotting?scriptVersionId=558039
- [ ] Report
	- [ ] Describing raw data (James)
	- [ ] Goal & methods/frameworks used (James)
	- [ ] Implementation/ steps (Philipp)
	- [ ] Results & difficulties (Philipp)

# Model Construction and Training (AJ and Ahmed)
- Important links
	- https://github.com/Lextal/pspnet-pytorch
- [x] UNet Implementation
- [x] Model training






# Papers
- [ ] A Review on Deep Learning Techniques Applied to Semantic Segmentation https://arxiv.org/pdf/1704.06857.pdf 
- [ ] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." CVPR. 2015. https://arxiv.org/pdf/1605.06211.pdf 
- [ ] He, Kaiming, et al. "Deep residual learning for image recognition." arXiv:1512.03385, 2015.
- [ ] Lee, Chen-Yu, et al. "Deeply-Supervised Nets." AISTATS, 2015. 
- [ ] Understanding Convolution for Semantic Segmentation, Wang et al
- [x] PSP Net https://arxiv.org/abs/1612.01105
- [ ] https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
- [ ] DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution and Fully Connected CRFs https://arxiv.org/pdf/1606.00915.pdf



# Overview of semantic segmentation over the years
* https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html and https://github.com/meetshah1995/pytorch-semseg 
