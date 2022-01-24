# Melanoma Classification Web App
This repository houses the code for a streamlit powered web app (capable of running on an AWS `t2.micro` EC2 instance) backed with a CNN fine-tuned on the [SIIM ISIC Melanoma Classification Competition](https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard) data.

![demo ss](images/demo_aws.png)

## Model Training:
- Note: Training code is NOT available in this repository.
- The CNN used is a `resnest50_fast_4s1x64d` variant of `ResNeSt` family of CNNs published in [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955) by Hang Zhang et al.
- The network is trained on an NVIDIA P100 TENSOR CORE GPU provided by [Kaggle](https://kaggle.com) in the GPU accelerator version of Kaggle Kernels, however the weight tensors are converted from cuda tensors to CPU tenesors to allow for inferring on machines without a GPU.
- The network is trained on two thirds of the ISIC 2020 JPEG images and all the JPEG images of ISIC 2019 (and 2018) resized to 128*128 sq. pixels for 15 epochs with a batch size of 256.
- The model reaches a validation AUC (calculated on the third part of ISIC 2020 data dropped from train set) of 0.8892 with single inference and 0.9010 with Test Time Augmentations.
- Providing better model weights is WIP.

## Getting Started:
Follow same steps to run the web app on a cloud vm.
- `git clone` this repository.
- [optional but recommended] Set up a virtual environement.
- Run `pip install -r requiments.txt` to install all* the python libraries.    
    - *`opencv-python` needs to be installed using [these](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) steps.
- Download the weights of the Neural Network from [here](https://drive.google.com/file/d/1M6w0WOjm1nLkDxl43_iwvIM-lVtfJPD_/view?usp=sharing). 
- Run `streamlit run app.py <path/of/the/weights_file(.ckpt)>` in the system CLI.
- In a web browser of choice, open `localhost:8501`.  

## Built with:
- **Python**
- **PyTorch: `torch` + `torchvision`**
- **Albumentations**
- **Numpy**
- **Streamlit**

## Author(s):
[Puneet Singh](https://twitter.com/p69ns)

## Acknowledgements:
- Zhang, H., Wu, C., Zhang, Z., Zhu, Y., Zhang, Z., Lin, H., Sun, Y., He, T., Muller, J., Manmatha, R., Li, M., & Smola, A. (2020). ResNeSt: Split-Attention NetworksarXiv preprint arXiv:2004.08955.
- Buslaev, A., Iglovikov, V., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. (2020). Albumentations: Fast and Flexible Image AugmentationsInformation, 11(2).
- Falcon, W. (2019). PyTorch LightningGitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning Cited by, 3.
***
***
