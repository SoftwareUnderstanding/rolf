# <p align="center"> GAN-replication </p>
Official code of ICCV 2021 [When does GAN replicate? An indication on the choice of dataset size](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_When_Do_GANs_Replicate_On_the_Choice_of_Dataset_Size_ICCV_2021_paper.pdf) by Qianli Feng, Chenqi Guo, Fabian Benitez-Quiroz, Aleix Martinez.

![GAN-REP results](./figures/github_image.png?raw=true)

# 1. BigGAN-PyTorch
This contains code for GPU training of BigGANs from Large Scale GAN Training for High Fidelity Natural Image Synthesis by Andrew Brock, Jeff Donahue, and Karen Simonyan.

This code is by Andy Brock and Alex Andonian.

![BIGGAN results](./figures/biggan.png?raw=true)

# 1.1. How To Use This Code
You will need:

PyTorch, version 1.0.1
tqdm, numpy, scipy, and h5py
The training set (for example, ImageNet)

First, you may optionally prepare a pre-processed HDF5 version of your target dataset for faster I/O. Following this (or not), you'll need the Inception moments needed to calculate FID. These can both be done by modifying and running:
```
./scripts/utils/prepare_data.sh
```
Which by default assumes your training set (images) is downloaded into the root folder ```data``` in this directory, and will prepare the cached HDF5 at 128x128 pixel resolution.

# 1.2. Metrics and Sampling
During training, this script will output logs with training metrics and test metrics, will save multiple copies (2 most recent and 5 highest-scoring) of the model weights/optimizer params, and will produce samples and interpolations every time it saves weights. The logs folder contains scripts to process these logs and plot the results using MATLAB.

After training, one can use ```sample.py``` to produce additional samples and interpolations, test with different truncation values, batch sizes, number of standing stat accumulations, etc. 

By default, everything is saved to weights/samples/logs/data folders.

# 1.3. An Important Note on Inception Metrics
This repo uses the PyTorch in-built inception network to calculate IS and FID. These scores are different from the scores you would get using the official TF inception code, and are only for monitoring purposes. Run sample.py on your model, with the ```--sample_npz``` argument, then run inception_tf13 to calculate the actual TensorFlow IS. Note that you will need to have TensorFlow 1.3 or earlier installed, as TF1.4+ breaks the original IS code.

# 1.4. 1-Nearest Neighbor Query
Here we provide 1-NN query on the original training image for each GAN generated image in 4 different latent space.

(a) To run 1-NN query in pixel-wise space:
```
python NN_query_thresh_finalVer.py
```
(b) To run 1-NN query in inceptionV3 space: for example
```
python NNquery_inceptionv3_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/biggan/NNquery_inception_v3/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32 
```

The figure below compared BigGAN image replications in RGB and InceptionV3 space:
![BIGGAN-NN-RGB-IncepV3 results](./figures/biggan_NN_rgb_incepv3.png?raw=true)

(c) To run 1-NN query in inceptionV3 concatenating pixel-wise space: for example
```
python NNquery_inceptionv3_pixelwise_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/biggan/NNquery_inceptionv3_pixelwise/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32
 ```
(d) To run 1-NN query in SimCLR space: for example
```
cd new_metrics_SimCLR
python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/BigGAN-PyTorch/imgs/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/biggan/NN_query/FLOWER_128_sub1000/Itr38950/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/biggan/NNquery_simCLR_v2/FLOWER_128_sub1000/Itr38950/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /Usr/data/flower/ \
 --old_batch_size 26
```

The figure below compared BigGAN image replications in RGB and SimCLR space:
![BIGGAN-NN-RGB-SimCLR results](./figures/biggan_NN_rgb_simclr.png?raw=true)

# 2. StyleGAN2 — Official TensorFlow Implementation
Analyzing and Improving the Image Quality of StyleGAN
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila

Paper: http://arxiv.org/abs/1912.04958

Video: https://youtu.be/c-NJtV9Jvp0

![STYLEGAN2 results](./figures/stylegan2.png?raw=true)

# 2.1. Requirements
* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
* On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
* Docker users: use the provided Dockerfile to build an image with the required library dependencies.

StyleGAN2 relies on custom TensorFlow ops that are compiled on the fly using NVCC. To test that your NVCC installation is working correctly, run:
```
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```
On Windows, the compilation requires Microsoft Visual Studio to be in PATH. We recommend installing Visual Studio Community Edition and adding into ```PATH``` using ```"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"```.

# 2.2. Preparing datasets
To generate TFRecord of the sub-dataset (for example, FLOWER_256_sub1000), use:
```
python dataset_tool.py \
 create_from_images datasets/FLOWER_256_sub1000 \
 /Usr/data/flower/jpg --resolution=256 \
 --random_subset=1000
```

# 2.3. Training networks
After TFRecord dataset (for example, FLOWER_256_sub1000) is created, run:
```
python run_training.py \
 --num-gpus=1 --data-dir=datasets \
 --config=config-f --dataset=FLOWER_256_sub1000 \
 --total-kimg=25000 --gamma=100 \
 --result-dir=results/results_FLOWER_256_sub1000
```
One can also add
```
--resume-pkl=results/results_FLOWER_256_sub1000/00001-stylegan2-FLOWER_256_sub1000-1gpu-config-f/network-snapshot-000322.pkl
```
to resume the training procedure.

# 2.4. 1-Nearest Neighbor Query
Here we provide 1-NN query on the original training image for each GAN generated image in 4 different latent space.

(a) To run 1-NN query in pixel-wise space:
```
python NN_getDist_testCode_forStylegan2.py
python NN_getRepThreshPairImg_testCode_forStylegan2.py
```
(b) To run 1-NN query in inceptionV3 space: for example
```
python NNquery_inceptionv3_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/stylegan2/datasets_images/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/ \
 --num_row 32 --num_col 32
```

The figure below compared StyleGAN2 image replications in RGB and InceptionV3 space:
![STYLEGAN2-NN-RGB-IncepV3 results](./figures/stylegan2_NN_rgb_incepv3.png?raw=true)

(c) To run 1-NN query in inceptionV3 concatenating pixel-wise space: for example
```
python NNquery_inceptionv3_pixelwise_myTest.py \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/stylegan2/datasets_images/FLOWER_128_sub1000/ \
 --gan_dir /Usr/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /Usr/gan_results_for_presentation/stylegan2/NNquery_inception_v3/FLOWER_128_sub1000_resume/fakes003248/ \
 --num_row 32 --num_col 32 
 ```
(d) To run 1-NN query in SimCLR space: for example
```
cd new_metrics_SimCLR
python NNquery_simCLR_myTest.py --model_path results_v2/FLOWER_128/best_128_0.5_200_26_2000_model.pth \
 --dataset FLOWER_128_sub1000 --data_dir /Usr/stylegan2/datasets_images/FLOWER_128_sub1000/ \
 --gan_dir /Usr/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/view_sampleSheetImgs/ \
 --result_dir /Usr/stylegan2/imgs/NNquery_simCLR_v2/FLOWER_128_sub1000_resume/fakes003248/ \
 --num_row 32 --num_col 32 --mean_std_data_dir /Usr/data/flower/ \
 --old_batch_size 26
```

The figure below compared StyleGAN2 image replications in RGB and SimCLR space:
![STYLEGAN2-NN-RGB-SimCLR results](./figures/stylegan2_NN_rgb_simclr.png?raw=true)

# 3. Dataset Complexity
To compute the Intrinsic Dimensionality (ID) for a dataset, run:
```
cd dataset_complexity
python intdim_mle_chenqi_v4.py
```

# 4. Fitting Dataset ID vs. GAN Replication Percentage Curves
To fit and plot the curves, run code in ```MATLAB_fit_plt_Curves/model/```.

Some fitting results are provided in ```MATLAB_fit_plt_Curves/results/```.

# 5. AMT Human Behavior Experiments
Codes and results are provided in ```humanBehavior_experiment/```.

# 6. Cite the paper
If this repository, the paper or any of its content is useful for your research, please cite:
```
@inproceedings{chenqiguo2021ganreplications,
      title={When does GAN replicate? An indication on the choice of dataset size}, 
      author={Qianli Feng and Chenqi Guo and Fabian Benitez-Quiroz and Aleix Martinez},
      booktitle={International Conference on Computer Vision (ICCV)},
      year={2021}
}
```


