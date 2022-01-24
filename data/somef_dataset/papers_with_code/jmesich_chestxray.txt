# Chest X-Ray (WIP)

### Purpose:  To create usable Chest Xrays with a GAN as a way to augment medical data without needing to worry about privacy and accessibility.

#### How: 
1. Create an implementation of DenseNet in PyTorch from : https://arxiv.org/abs/1608.06993
2. Create an implementation of StyleGan in PyTorch from : https://github.com/NVlabs/stylegan
3. Generate images from StyleGan
4. Train the DenseNet two different times, with the real images and the StyleGan generated images
5. Compare the scores

#### What data is being used:
All the data was used can be found here: https://www.kaggle.com/nih-chest-xrays/data

#### Technologies: PyTorch, StyleGAN, DenseNet

Tasks:
| Task  | Complete  |
|---|---|
| Gather Data  |  :heavy_check_mark: |
| Code DenseNet  | :heavy_check_mark:  |
| Code StyleGan  |   |
| Generate images  |   |
| Train two DenseNets  |   |
| Analyze results  |   |
