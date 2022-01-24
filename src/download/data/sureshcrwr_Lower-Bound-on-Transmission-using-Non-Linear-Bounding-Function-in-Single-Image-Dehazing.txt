# Lower-Bound-on-Transmission-using-Non-Linear-Bounding-Function-in-Single-Image-Dehazing
Lower Bound on Transmission Using Non-Linear Bounding Function in Single Image Dehazing

The visibility of an image captured in poor weather (such as haze, fog, mist, smog) degrades due to scattering of light by atmospheric particles. Single image dehazing (SID) methods are used to restore visibility from a single hazy image. The SID is a challenging problem due to its ill-posed nature. Typically, the atmospheric scattering model (ATSM) is used to solve SID problem. The transmission and atmospheric light are two prime parameters of ATSM. The accuracy and effectiveness of SID depends on accurate value of transmission and atmospheric light. The proposed method translates transmission estimation problem into estimation of the difference between minimum color channel of hazy and haze-free image. The translated problem presents a lower bound on transmission and is used to minimize reconstruction error in dehazing. The lower bound depends upon the bounding function (BF) and a quality control parameter. A non-linear model is then proposed to estimate BF for accurate estimation of transmission. The proposed quality
control parameter can be utilized to tune the effect of dehazing. 

More details on Non-Linear-Bounding-Function can be found in the [paper](https://doi.org/10.1109/TIP.2020.2975909) (or [this link](https://ieeexplore.ieee.org/document/9018379)) titled as _Lower-Bound-on-Transmission-using-Non-Linear-Bounding-Function-in-Single-Image-Dehazing._ (Published in Transaction on Image Processing)

# Our Environment
- Windows 10
- Matlab 2017a

# Steps to perform Dehazing
1. Upload hazy images in the folder, named "Data"

2. Run Start_Dehazing.m
3. Respective haze-free images will be saved in the folder, named Results

# Visual Results
![image](https://user-images.githubusercontent.com/71458796/133307037-7418df7e-13e2-4470-bdd7-1c490b7e8d9b.png)

# Quantitative Results
![image](https://user-images.githubusercontent.com/71458796/133307122-61c67485-e205-487a-8e96-20ef0d2a6104.png)

# Computational Complexity
![image](https://user-images.githubusercontent.com/71458796/133307291-544147e0-bc51-4500-b017-a6409b03b621.png)

![image](https://user-images.githubusercontent.com/71458796/133307372-cd919f88-8435-4feb-8e1d-86df0cb2bef5.png)

# Remarks
1. This code is based on [Non-Linear Bounding Function](https://doi.org/10.1109/TIP.2020.2975909)
2. This code is for research and study purpose. If you use this code, then cite the follwing paper.

# Citation (Plain Text)
S. C. Raikwar and S. Tapaswi, "Lower Bound on Transmission Using Non-Linear Bounding Function in Single Image Dehazing," in IEEE Transactions on Image Processing, vol. 29, pp. 4832-4847, 2020, doi: 10.1109/TIP.2020.2975909.

# Citation (BibTex)
@article{raikwar-tapaswi,
  author={Raikwar, Suresh Chandra and Tapaswi, Shashikala},
  journal={IEEE Transactions on Image Processing}, 
  title={Lower Bound on Transmission Using Non-Linear Bounding Function in Single Image Dehazing}, 
  year={2020},
  volume={29},
  number={},
  pages={4832-4847},
  doi={10.1109/TIP.2020.2975909}}
