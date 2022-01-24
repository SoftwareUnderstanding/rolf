
# Pneumothorax Identification from X-Ray Images 

 [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-grey?style=for-the-badge&logo=Google-Colab)](https://githubtocolab.com/reyvaz/pneumothorax_detection/blob/master/notebooks/pneumothorax_detection.ipynb) 
 
<br>

This repo contains my solution to the [SIIM-ACR Pneumothorax Segmentation; Identify Pneumothorax Disease in Chest X-Rays](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) challenge. The challenge consists on detecting pneumothorax disease (i.e. collapsed lung disease) from x-ray images and masking the location of the disease within the x-ray image. 

##### The solution consists on a two-step approach. 

1. The first step is to run the images through an ensemble of EfficientNet (Tan & Le 2020) based binary image classifiers to determine whether the x-ray presents pneumothorax disease. 

2. The second step, runs the image through an ensemble of Unet (Ronneberger et al., 2015) and Unet ++ (Zhou et al., 2019) networks with EfficientNet backbones to identify the location of the disease. 

This approach achieves a [Dice coefficient of 0.8522](https://www.kaggle.com/reyvaz/pneumothorax-inference-submission?scriptVersionId=59143042) in the offcial private test data of the competition, on par with the top 3% of the results. 

**Re-running the Notebook**

- [Open the notebook in Colab](https://githubtocolab.com/reyvaz/pneumothorax_detection/blob/master/notebooks/pneumothorax_detection.ipynb) and select TPU as accelerator (recommended).
- Update the GCS Path as indicated in the notebook
- Run all

#### About the Dataset

To train using TPUs, I extracted the original data from DICOM files (X-Rays, patient metadata) and CSV files (mask RLE encodings) and placed it in TFRec files. The original DICOM and CSV data can be found [here](https://www.kaggle.com/seesee/siim-train-test). The TFRecs used in the notebook can be found [here](https://www.kaggle.com/reyvaz/siimacr-pneumothorax-segmentation-tfrecs).

#### Acknowledgements:

Thanks to [The Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/) and the [American College of Radiology (ACR)](https://www.acr.org/) for creating and providing the dataset.

#### References:

- Ronneberger O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. [arXiv:1505.04597v1](https://arxiv.org/abs/1505.04597v1).
- Tan, M., & Le, Q. V. (2020). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [arXiv:1905.11946v5](https://arxiv.org/abs/1905.11946v5).
- Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. [arXiv:1912.05074v2](https://arxiv.org/abs/1912.05074v2).

