﻿﻿﻿﻿﻿# 论文复现：Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation 

*****

[English](README.md)|[简体中文](README_cn.md)

* paddle-PSP
  * [1 Introduction](#1-Introduction)
  * [2 Result](#2-Result)
  * [3 Datasets](#3-Datasets)
  * [4 Environment](#4-Environment)
  * [5 Pretrained models](#5-Pretrained-models)
  * [6 Quick start](#6-Quick-start)
    * [train](#train)
    * [inferernce](#inferernce)
  * [7 Code structure](#7-Code-structure)
    * [structure](#structure)
    * [Parameter description](#Parameter-description)
  * [8 Model information](#8-Model-information)

# 1 Introduction

***

This project is based on the pixel2style2pixel (pSp). pSp framework generates a series of style vectors based on a novel encoder network, which is fed into a pre-trained style generator to form an extended W + potential space. The encoder can directly reconstruct real input images.

#### Paper

* [1] Richardson E ,  Alaluf Y ,  Patashnik O , et al. Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation[J].  2020.

#### Reference project

* [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

#### **Project on Ai Studio**

* notebook：[https://aistudio.baidu.com/aistudio/projectdetail/2331440](https://aistudio.baidu.com/aistudio/projectdetail/2331440)

# **2 Result**

#### **Results（Test on CelebA-HQ）**

|  Model  | LPIPS | Similarity | MSE  |
| :-----: | :---: | :--------: | :--: |
|  Paper  | 0.17  |    0.56    | 0.03 |
| Pytorch | 0.15  |    0.57    | 0.03 |
| Paddle  | 0.17  |    0.57    | 0.03 |

#### **Visual comparison**

|       Pytorch                                           |                           Paddle                             |
| :-----------------------------------------------------: | :----------------------------------------------------------- |
|  <img src="examples/1.png" alt="1" style="zoom:100%;" /> | <img src="inference/inference_coupled/052329.jpg" alt="1" style="zoom: 25%;" />                 |
| <img src="examples/2.png" alt="1" style="zoom:100%;" /> | <img src="inference/inference_coupled/179349.jpg" alt="1" style="zoom: 25%;" /> |
| <img src="examples/3.png" alt="1" style="zoom:100%;" /> | <img src="inference/inference_coupled/145789.jpg" alt="1" style="zoom:25%;" /> |

# **3 Datasets**

- Training： [FFHQ-1024](https://github.com/NVlabs/ffhq-dataset).  saved in `FFHQ/`.


- Testing：[CelebA-HQ](https://aistudio.baidu.com/aistudio/datasetdetail/49226).saved in`CelebA_test/`.


# 4 Environment

Hardware：GPU、CPU

Framework：PaddlePaddle >=2.0.0

# 5 Pretrained models

Pretrained models saved in`pretrained_models/`.

| Pretrained models                                         | Description                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| FFHQ StyleGAN(stylegan2-ffhq-config-f.pdparams)           | StyleGAN trained with the FFHQ dataset from[rosinality](https://github.com/rosinality/stylegan2-pytorch) ，output size:1024x1024 |
| IR-SE50 Model(model_ir_se50.pdparams)                     | IR_SE model ([TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch))trained for computering ID loss. |
| CurricularFace Backbone(CurricularFace_Backbone.paparams) | Pretrained CurricularFace model([HuangYG123](https://github.com/HuangYG123/CurricularFace))evaled Similarity |
| AlexNet(alexnet.pdparams和lin_alex.pdparams)              | computered lpips loss                                        |
| StyleGAN Inversion(psp_ffhq_inverse.pdparams)             | pSp trained with the FFHQ dataset for StyleGAN inversion.    |

Baidu driver：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg) password：m3nb

Pretrained pSp encoder：

| 模型                                          | Description                                               |
| --------------------------------------------- | --------------------------------------------------------- |
| StyleGAN Inversion(psp_ffhq_inverse.pdparams) | pSp trained with the FFHQ dataset for StyleGAN inversion. |

# 6 Quick start

#### Compile operation

	python scripts/compile_ranger.py

#### Train

	python scripts/train.py \
	--dataset_type=ffhq_encode \
	--exp_dir=exp/test \
	--workers=0 \
	--batch_size=8 \
	--test_batch_size=8 \
	--test_workers=0 \
	--val_interval=2500 \
	--save_interval=5000 \
	--encoder_type=GradualStyleEncoder \
	--start_from_latent_avg \
	--lpips_lambda=0.8 \
	--l2_lambda=1 \
	--id_lambda=0.1 \
	--optim_name=ranger

#### inferernce

```
python scripts/inference.py \
--exp_dir=inference \
--checkpoint_path=pretrained_models/psp_ffhq_inverse.pdparams \
--data_path=CelebA_test \
--test_batch_size=8 \
--test_workers=4
```

#### Others

* LPIPS

```
python scripts/calc_losses_on_images.py \
--mode lpips \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```

* MSE

```
python scripts/calc_losses_on_images.py \
--mode l2 \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```

* Similarity

```
python scripts/calc_id_loss_parallel.py \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```

# 7 Code structure

#### **Structure**

```
├─config          # 配置
    ├─data            #数据集加载
       ├─CelebA_test  # 测试数据图像
    ├─logs            #日志
       ├─train        # 训练日志
       ├─test         # 测试日志
    ├─models          # 模型
        ├─encoders    # 编码器
        ├─loss        # 损失函数
        ├─mtcnn       #     
        ├─stylegan2   #       
        ├─utils       # 编译算子
    ├─scripts         #算法执行
        trian         #训练
        inference     #测试
    ├─utils           # 工具代码
    │  README.md      #英文readme
    │  README_cn.md   #中文readme
```

#### **Parameter description**

| Parameter | Default |
| --------- | ------- |
| Config    | None    |

# 8 Model information

The overall information of the model is as follows:

| Information | Descriptions     |
| ----------- | ---------------- |
| Version     | Paddle 2.1.2     |
| Application | Image Generation |
| Hardware    | GPU / CPU        |

# License

```
# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

