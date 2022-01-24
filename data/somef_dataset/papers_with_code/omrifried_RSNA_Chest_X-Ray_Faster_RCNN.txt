# RSNA_XRay_Faster_RCNN

PyTorch implementation of Faster RCNN with a ResNet-50-FPN backbone on the RSNA Chest X-Ray dataset.

The RSNA Chest X-Ray dataset, available from [kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/acknowledgements), contains a
total of 30,2257 1024 x 1024 chest x-ray images from 26,684 unique patients. The patients fall into one of three categories: Lung Opacity, Normal, or
No Lung Opacity / Not Normal. 

For this model, only patients with Lung Opacity were included in order to predict the bounding box of the opacity. Of the 30,2257 images, approximately 10,000 images contained opacity. Furthermore, the 10,000 images belong to a total of 6,012 unique patients. The bounding boxes were combined to be on a per-patient basis
such that certain images could contain multiple boxes. 

Due to limited capacity, the model was only trained for a small number of epochs. If the model is trained for a longer
amount of time, improved results should be available.


Citations:<br />
1. Rui P, Kang K. National Ambulatory Medical Care Survey: 2015 Emergency Department Summary Tables. Table 27. Available from: www.cdc.gov/nchs/data/nhamcs/webtables/2015edwebtables.pdf

2. Deaths: Final Data for 2015. Supplemental Tables. Tables I-21, I-22. Available from: www.cdc.gov/nchs/data/nvsr/nvsr66/nvsr6606tables.pdf

3. Franquet T. Imaging of community-acquired pneumonia. J Thorac Imaging 2018 (epub ahead of print). PMID 30036297

4. Kelly B. The Chest Radiograph. Ulster Med J 2012;81(3):143-148

5. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017, http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf

6. P. Rajpurkar et al. (Dec. 2017). ‘‘CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning.’’ [Online]. Available:
https://arxiv.org/abs/1711.05225
