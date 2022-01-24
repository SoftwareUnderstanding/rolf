
# Steel Defect Detection, Classification, and Segmentation

 [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-grey?style=for-the-badge&logo=Google-Colab)](https://githubtocolab.com/reyvaz/steel-defect-segmentation/blob/master/steel_defect_detection.ipynb) 

<br>

This notebooks presents a solution to the [Severstal: Steel Defect Detection]( https://www.kaggle.com/c/severstal-steel-defect-detection) competition. The challenge consists on using images to detect defects on pieces of steel, classify the type of defect (4 types), and identify the location and area of the defect.

This solution consists on a two-step approach

1. The first step is an ensemble of binary image neural network classifiers to determine whether the piece of steel in the image presents a  defect.

2. The second step is an ensemble of segmentation neural networks to identify the location of the defect and identify the type of the defect.

### Training

All 1st and 2nd step neural networks were trained on a K = 5, K-Fold cross-validation distribution of the data, all with the same image size. Random data augmentations were applied to the training partition data for all networks and folds in both stages.

### Binary Classification

The ensemble for the binary classification step consists of EfficientNet (Tan & Le 2020) based classifiers versions B0-B5. 

### Defect Type Classification and Segmentation 

The ensemble for the 2nd step consists of UNet++ (Zhou et al., 2019) based CNNs, all with EfficientNet backbones versions B0-B5. 

**Running the Notebook**

- [Open the notebook in Colab](https://githubtocolab.com/reyvaz/steel-defect-segmentation/blob/master/steel_defect_detection.ipynb) and select TPU as accelerator.
- Update the GCS Path as indicated in the notebook
- Run all

#### References:

- Tan, M., & Le, Q. V. (2020). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [arXiv:1905.11946v5](https://arxiv.org/abs/1905.11946v5).

- Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. [arXiv:1912.05074v2](https://arxiv.org/abs/1912.05074v2).