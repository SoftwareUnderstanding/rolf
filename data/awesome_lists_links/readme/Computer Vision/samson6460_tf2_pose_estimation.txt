# tf2_pose_estimation

![example](https://img.shields.io/badge/Python-3.x-blue.svg) ![example](https://img.shields.io/badge/Tensorflow-2.x-yellow.svg)

It's a framework of Pose Estimation implemented in tensorflow 2.x.

There're Stacked Hourglass Network、DeepLabV3+、U-Net in this framework.

**Stacked Hourglass Network**: Stacked Hourglass Networks for Human Pose Estimation by Alejandro Newell, Kaiyu Yang, Jia Deng. (https://arxiv.org/abs/1603.06937).

![Stacked Hourglass](images/stacked-hg.png)
(from https://www.arxiv-vanity.com/papers/1603.06937/)

**DeepLabV3+**: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation by Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam (https://arxiv.org/abs/1802.02611).

![DeepLabV3+](https://imgs.developpaper.com/imgs/2068779560-9ebbf4c31e6837f3_articlex.png)

**U-Net**: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, Thomas Brox (https://arxiv.org/abs/1505.04597).

![U-Net](images/modified_unet.png)

**ResUNet**: A U-net like model with ResNet backbone and UpResBlock.

![ResUNet](images/ResUNet.png)

# Table of Contents

- [tf2_pose_estimation](#tf2_pose_estimation)
- [Table of Contents](#table-of-contents)
- [Usage](#usage)

# Usage

1. Clone or download
    - Use the command bellow in terminal to git clone:    
    ```git clone https://github.com/samson6460/tf2_pose_estimation.git```

    - Or just download whole files using the **[Code > Download ZIP]** button in the upper right corner.
    
2. Install dependent packages: 
    ```pip install -r requirements.txt```

3. Import tf2_pose_estimation:
   ```import tf2_pose_estimation```