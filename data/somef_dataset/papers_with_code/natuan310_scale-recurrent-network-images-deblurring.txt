CoderSchool - Machine Learning Engineer
Final Project - Nguyen Anh Tuan

# Image Deblurring by Scale - Recurrent Network


## 1. INTRODUCTION

Nowadays, we have so many high resolution camera, from camera to smartphone, every devices can take photos or videos at high resolution. But sometime we miss a great moments because of blurring or we want to get a photo from video but all the frames is blury.

This project is to build an web-application use Scale-Recurrent Network for deblur that blurred images to restore the value that we want to get from that images.

## 2. DATASET

We will need data is blurry and sharp images to train the model. GOPRO has a dataset which contains sharp and blurry-by-AI images extracts from street-record videos.

You can download a light version (9GB) or the complete version (35GB).

[GOPRO LIGHT 9GB](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view)

[GOPRO FULL 35GB](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view)

[Images from Mobile Images](https://www.kaggle.com/kwentar/blur-dataset)


## 3. Scale - Recurrent Network

#### 4a. Network Architecture
![](https://i.imgur.com/277m6sj.png)

The overall architecture of the proposed network, which we call SRN-DeblurNet. It takes as input a sequence of blurry images downsampled from the input image at different scales, and produces a set of corresponding sharp images. The sharp one at the full resolution is the final output.

#### 4b. Scale-recurrent Network (SRN)
We adopt a novel recurrent structure across multiple scales in the coarse-to-fine strategy. We form the generation of a sharp latent image at each scale as a sub-problem of the image deblurring task, which takes a blurred image and an initial deblurring result (upsampled from previous scale) as  input, and estimates the sharp image at this scale as:
<p align="center">
  <img height="80px" src="https://i.imgur.com/FcX97PD.png">
</p>

where i is the scale index, with i = 1 representing the finest scale. $B^{i}$ and $I^{i}$ are the blurry and estimated latent images at the i-th scale, respectively. $Net_{SR}$ is our proposed scalerecurrent network with training parameters denoted as $θ_{SR}$.
Since the network is recurrent, hidden state features hi flow across scales. The hidden state captures image structures and kernel information from the previous coarser scales. $(·)^{↑}$ is the operator to adapt features or images from the (i + 1)-th to i-th scale.

Eq. (1) gives a detailed definition of the network. In practice, there is enormous flexibility in network design.

First, recurrent networks can take different  orms, such as vanilla RNN, long-short term memory (LSTM) and gated recurrent unit (GRU). We choose ConvLSTM since it performs better in our experiments.

Second, possible choices for operator (·)↑ include deconvolution layer, sub-pixel convolution layer and image resizing. We use bilinear interpolation for all our experiments for its sufficiency and simplicity.

Third, the network at each scale needs to be properly designed for optimal effectiveness to recover the sharp image. Our method is detailed in the following.

#### 4c. Encoder-Decoder with Resblocks

**Encoder-decoder Network**
Encoder-decoder network refers to the symmetric CNN structures that first progressively transform input data into feature maps with smaller spatial sizes and more channels (in encoder), and then transform them back to the shape of the input (in decoder). Skip-connections between corresponding feature maps are widely used to combine different levels of information. They can also benefit gradient propagation and accelerate convergence. Typically, the encoder contains several stages of convolution layers with strides, and the decoder module is implemented using a series of deconvolution layers or resizing. Additional convolution layers are inserted after each level to further increase depth.

The encoder-decoder structure has been proven to be effective in many vision tasks. However, directly using the encoder-decoder network is not the best choice for our task with the following considerations. First, for the task of deblurring, the receptive field needs to be large to handle severe motion, resulting in stacking more levels for encoder/decoder modules. However, this strategy is not recommended in practice since it increases the number of parameters quickly with the large number of intermediate feature channels. Besides, the spatial size of middle feature map would be too small to keep spatial information for reconstruction. Second, adding more convolution layers at each level of encoder/decoder modules would make the network slow to converge (with flat convolution at each level). Finally, our proposed structure requires recurrent modules with hidden states inside.

**Encoder/decoder ResBlock** 
We make several modifications to adapt encoder-decoder networks into our framework. First, we improve encoder/decoder modules by introducing residual learning blocks. Based on results of [25] and our experiments, we choose to use ResBlocks instead of the original one in ResNet [15] (without batch normalization). As illustrated in Fig. 3, our proposed Encoder ResBlocks (EBlocks) contains one convolution layer followed by several ResBlocks. The stride for convolution layer is 2. It doubles the number of kernels of previous layer and downsamples the feature maps to half size. Each of the following ResBlocks contains 2 convolution layers. Besides, all convolution layers have the same number of kernels. Decoder ResBlock (DBlocks) is symmetric to EBlock.

It contains several ResBlocks followed by one deconvolution layer. The deconvolution layer is used to double the spatial size of feature maps and halve channels.

Second, our scale-recurrent structure requires recurrent modules inside networks. Similar to the strategy of [35], we insert convolution layers in the bottleneck layer for hidden state to connect consecutive scales. Finally, we use large convolution kernels of size 5×5 for every convolution layer.

The modified network is expressed as
<p align="center">
  <img src="https://i.imgur.com/mg2gw6t.png">
</p>
where $Net_{E}$ and $Net_{D}$ are encoder and decoder CNNs with parameters $θ_{E}$ and $θ_{D}$. 3 stages of EBlocks and DBlocks are used in $Net_{E}$ and $Net_{D}$, respectively. $θ_{LSTM}$ is the set of parameters in ConvLSTM. Hidden state $h^{i}$ may contain useful information about intermediate result and blur patterns, which is passed to the next scale and benefits the fine-scale problem.

The details of model parameters are specified here. Our SRN contains 3 scales. The (i + 1)-th scale is of half size of the i-th scale. For the encoder/decoder ResBlock network, there are 1 InBlock, 2 EBlocks, followed by 1 Convolutional LSTM block, 2 DBlocks and 1 OutBlock, as shown in Fig. 3.
InBlock produces a 32-channel feature map. OutBlock takes previous feature map as input and generates output image. The numbers of kernels of all convolution layers inside each EBlock/DBlock are the same.

For EBlocks, the numbers of kernels are 64 and 128, respectively. For DBlocks, they are 128 and 64. The stride size for the convolution layer in EBlocks and deconvolution layers is 2, while all others are 1. Rectified Linear Units (ReLU) are used as the activation function for all layers, and all kernel sizes are set to 5.

#### 4d. Eucliden Loss Function
We use Euclidean loss for each scale, between network output and the ground truth (downsampled to the same size using bilinear interpolation) as
<p align="center">
  <img height="100px" src="https://i.imgur.com/FEBtjRY.png">
</p>
where $I^{i}$ and $I^{i}_{∗}$ are our network output and ground truth respectively in the $i^{-th}$ scale. $κ_{i}$ are the weights for each scale. We empirically set $κ_{i}$ = 1.0. $N_{i}$ is the number of elements in $I^{i}$ to normalize. We have also tried total variation and adversarial loss. But we notice that L2-norm is good enough to generate sharp and clear results.

## 5. Training Process
#### 5a. Optimizer
For model training, we use Adam solver with $β_{1}$ = 0.9, $β_{2}$ = 0.999 and  = $10^{−8}$. The learning rate is exponentially decayed from initial value of 0.0001 to $1e^{−6}$ at 2000 epochs using power 0.3. According to our experiments, 2,000 epochs are enough for convergence, which takes about 72 hours. In each iteration, we sample a batch of 16 blurry images and randomly crop 256 × 256-pixel patches as training input. Ground truth sharp patches are generated accordingly. All trainable variables are initialized using Xavier method. The parameters described above are fixed for all experiments.

For experiments that involve recurrent modules, we apply gradient clip only to weights of ConvLSTM module (clipped by global norm 3) to stabilize training. Since our network is fully convolutional, images of arbitrary size can be fed in it as input, as long as GPU memory allows. For a testing image of size 720 × 1280, running time of our proposed method is around 1.87 seconds.

#### 5b. Metric Monitoring
**Peak Signal-to-Noise Ratio (PSNR)**
PSNR is most easily defined via the mean squared error (MSE). Given a noise-free m×n monochrome image I and its noisy approximation K, MSE is defined as:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/3a34719b4f391dba26b3e8e4460b7595d62eece4">

The PSNR (in dB) is defined as:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fc22801ed1232ff1231c4156b589de5c32063a8a">

Here, MAXI is the maximum possible pixel value of the image. When the pixels are represented using 8 bits per sample, this is 255. More generally, when samples are represented using linear PCM with B bits per sample, MAXI is 2B−1.


**Structural Similarity (SSIM)**
The structural similarity (SSIM) index is a method for predicting the perceived quality of digital television and cinematic pictures, as well as other kinds of digital images and videos. The basic model was developed in the Laboratory for Image and Video Engineering (LIVE) at The University of Texas at Austin and further developed jointly with the Laboratory for Computational Vision (LCV) at New York University. Further variants of the model have been developed in the Image and Visual Computing Laboratory at University of Waterloo and have been commercially marketed.

SSIM is used for measuring the similarity between two images. The SSIM index is a full reference metric; in other words, the measurement or prediction of image quality is based on an initial uncompressed or distortion-free image as reference. SSIM is designed to improve on traditional methods such as peak signal-to-noise ratio (PSNR) and mean squared error (MSE).

The difference with respect to other techniques mentioned previously such as MSE or PSNR is that these approaches estimate absolute errors; on the other hand, SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. Structural information is the idea that the pixels have strong inter-dependencies especially when they are spatially close. These dependencies carry important information about the structure of the objects in the visual scene. Luminance masking is a phenomenon whereby image distortions (in this context) tend to be less visible in bright regions, while contrast masking is a phenomenon whereby distortions become less visible where there is significant activity or "texture" in the image.

The SSIM index is calculated on various windows of an image. The measure between two windows x and y of common size N×N is:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/63349f3ee17e396915f6c25221ae488c3bb54b66">


## 6. Training Results
![](https://i.imgur.com/d2xaKI9.jpg)

![](https://i.imgur.com/78zAKJ1.jpg)

![](https://i.imgur.com/hTHKJMM.jpg)

![](https://i.imgur.com/4AS9XJT.jpg)

## 7. Deblur Result


## TO-DO LIST

- [x] 1. Get dataset.
- [X] 2. Research about Deblur Images by Neural Network.
- [X] 3. Design the model base on research.
- [X] 4. Preprocess data.
- [X] 5. Traning and validating model.
- [X] 6. Build the flask app and deploy model.

## RESOURCE
https://www.sicara.ai/blog/2018-03-20-GAN-with-Keras-application-to-image-deblurring
https://deepai.org/publication/blind-deblurring-using-gans
http://openaccess.thecvf.com/content_ICCV_2019/papers/Kupyn_DeblurGAN-v2_Deblurring_Orders-of-Magnitude_Faster_and_Better_ICCV_2019_paper.pdf
https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
https://arxiv.org/pdf/1704.00028.pdf
https://github.com/jiangsutx/SRN-Deblur
https://arxiv.org/abs/1802.01770
https://jleinonen.github.io/2019/11/07/gan-elements-2.html
https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e
https://en.wikipedia.org/wiki/Residual_neural_network
https://en.wikipedia.org/wiki/Long_short-term_memory
https://en.wikipedia.org/wiki/Recurrent_neural_network
https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
https://en.wikipedia.org/wiki/Structural_similarity
