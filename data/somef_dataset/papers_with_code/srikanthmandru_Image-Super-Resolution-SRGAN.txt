
# Title: Super Resolution of images/videos using SRGAN

#### Authors : Srikanth Babu Mandru

#### *Proposed as "Quantiphi Project"*

### New update on project: Modified the code to TensorFlow 2.x version compatible and other APIs. Please, look at file starting with "TF2_" and make respective changes to loss function in order to work with different losses. Also, the cross-platform mobile application for this project is included as separate repository @[Application link]( https://github.com/srikanthmandru/TFLite-Flutter-Super-Resolution-App ). Working on multiple ideas and code for that will not be available here until I publish a paper. (Please, contact me for details on research work!!!)

## Summary 

Super-resolution (SR) of images refers to the process of generating or reconstructing the high- resolution (HR) images from low-resolution images (LR). This project mainly focuses on dealing with this problem of super-resolution using the generative adversarial network, named SRGAN, a deep learning framework. In this project, SRGAN was trained and evaluated using 'DIV2K', ‘MS-COCO’ and ‘VID4’ [6] which are the popular datasets for image resolution tasks.

In total, datasets were merged to form: 

1. 5800 training images

2. 100 validation images

3. 4 videos for testing

Apart from the datasets mentioned above, ‘LFW’, ‘Set5’ and ‘Set14’ datasets [6] were used to get inferences and compare the performance of models implemented in this project with the models from Ledig et al. [2].

Most of this project is built upon the ideas of Ledig et al [2]. Apart from that, I did some research on comparing the results obtained using different objective functions available in TensorFlow’s “TFGAN” library for loss optimizations of SRGAN. Different model implementations were evaluated for pixel quality through the peak signal-to-noise ratio (PSNR) scores as a metric. Intuitively, this metric does not capture the essence of the perceptual quality of an image. However, it is comparatively easy to use PSNR when evaluating the performance while training the model compared to mean-opinion-score (MOS) that has been used by Ledig et al [2]. To evaluate the perceptual quality of images, I have compared the generated images from both the models. This paper also proposes a method of super-resolution using SRGAN with “Per-Pix loss” which I defined in the losses section of this paper. Based on results from [2] and [5], I have combined both MSE and VGG losses, named it “Per-Pix loss” that stands for ‘Perceptual and Pixel’ qualities of the image, which resulted in preserving the pixel quality besides improving the perceptual quality of images. Finally, I have compared the models built in this project with the models from Ledig et al. [2] to know the performance and effectiveness of models implemented in this project.

## Proposed plan of research

In first phase of this project, I have implemented the SRGAN which is a GAN-based model using TensorFlow, Keras and other Machine learning APIs. I choose Peak signal-to-noise-ratio [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) as the key metric to evaluate the model's performance. Proposed a new loss, namely 'Per-Pix' loss, for SRGAN model training and observed significant improvement in PSNR values with fewer iterations of training compared to model trained with 'Perceptual Loss'.

Now, in second phase of this project, I pickup from the first phase results and focus on comparing the model performances trained separately with 'Per-Pix', 'Perceptual' and 'MSE' losses through 'PSNR' metric. Apart from this, I will do research on using various other model architectures. There is also a great need for proper metric to evaluate the image quality. For this, currently, I found the paper [8] which detailed about different metrics that can be used for evaluating the image resolution quality. In this paper, they have described how various metrics are related to the perceptual quality. So, I will study further on papers [3] and [8] to get deeper understanding and arrive at right approaches in order to solve super-resolution problem. If I find any reasonable approach or ideas that would impact the performance, I will incorporate those into the project.

Coming to training stage of this project, it requires a huge effort to train these massive models. Thus, all of the training will be done using the Google cloud platform (GCP) AI services and products. During training, I make use of NVIDIA Tesla P100 GPUs with CUDA and cuDNN toolkits to leverage faster training offered by GPUs. As a part of training procedure, I will also create a visualization dashboard consisting of model architecture and results using TensorBoard to better interpret the results while training is in progress. After the training stage, the model will be deployed using google cloud AI platform for future predictions. Also, the best model will be used to super-resolve the videos using a new data pipeline to process the videos. Further, I am planning to deploy the model as an application to real-world users using TensorFlow Lite. Overall, in phase 2, I primarily concentrate on training and deploying the SRGAN model besides doing further research.


## Results

Initially, I have implemented the image preprocessing part of project so that images data fits to our model seamlessly. The steps that I have followed are as follows :

- The actual original images are of size (2048, 1080, 3)

- Cropped the original images to size (256, 256, 3)

- Now, discriminator input will be of shape (256, 256, 3) and generator input is (64, 64, 3) which is downsampled version of discriminator input using "bicubic" kernel with factor of "4"   

Some of the sample low and high resolution images that are obtained from image preprocessing stage are as shown in below figure:

<img src ="downloaded images/image_preprocess/low_res1.png" width = "400" height = "400" /> <img src ="downloaded images/image_preprocess/high_res1.png" width = "400" height = "400" /> 

Models implemented were evaluated in terms of "Pixel quality" and "Perceptual quality" through "PSNR" and "Visualization of Images". 

### Pixel quality evaluation:

To better understand how efficient the models perform, I have compared the results of models implemented in this project with corresponding model results from Ledig et al. [2] using the same datasets, that is, Set5 and Set14 datasets and results were tabulated in the below figure. While comparing, MSE and PERPIX trained models in this project are compared with MSE and VGG54 models from Ledig et al. [2] respectively.

From the tables of below figure, we can observe that the models implemented here in this project are performing well-enough considering the number of training steps that models have been trained for. PSNR values obtained were close to the results from Ledig et al. [2] and with further training, these values will improve. From the below figure, another noticeable point is that PSNR value was slightly dropped between MSE and VGG54 trained models (from Ledig et al. [2]) on both datasets. With the PERPIX loss, since we are preserving the pixel quality through the MSE loss component besides the VGG component, we can observe a slight increase in PSNR between MSE and PERPIX trained models (in this project) on both datasets. Thus, the SRGAN model trained with PERPIX loss can be considered as the better choice over the VGG54 (from Ledig et al. [2]) in terms of pixel quality that we have measured through PSNR.

<img src ="downloaded images/psnr_results/set5_comparison.png" width = "400" height = "250" /> <img src ="downloaded images/psnr_results/set14_comparison.png" width = "400" height = "250" /> 

<p align="center"> Figure: Comparison tables of PSNR values (in each table cell) on Set5 and Set14 datasets <p align="center">

### Perceptual quality evaluation:

To evaluate the perceptual quality of images, I have compared the images generated from both the models with corresponding down-sampled low- resolution and the original high-resolution images. Images for comparison were included below. we can observe that both SRGAN models are generating better quality images (visually) from the respective down-sampled low- resolution images. However, there is a noticeable difference between generated images and the original high-resolution image.

***Note:*** In the case of PERPIX trained model, this difference can be reduced with further iterations as it can be observed from Ledig et al. [2] that it took almost 2 ∗ 105 training steps to reach notifiable perceptual quality. We can also observe from Ledig et al. [2] that it took 20k iterations to actually diverge from the pre-trained SR-ResNet model (trained with MSE loss) and start learning the high-frequency details. Thus, ***FURTHER TRAINING IS REQUIRED***.

<img src ="downloaded images/sets_results_images/set5 results/result_images1.png" width = "800" height = "300" /> 

<img src ="downloaded images/sets_results_images/set5 results/result_images2.png" width = "800" height = "300" /> 

<img src ="downloaded images/sets_results_images/set14 results/result_images8.png" width = "800" height = "300" /> 

<p align="center"> Figure: Down-sampled image, Original image, generated images from MSE and PERPIX trained models (from left to right) <p align="center">


#### Close Observation of images: (On validation data)

From figures below, it can be observed that the generated images from MSE trained model are consisting of pixelated boxes, whereas PERPIX trained model’s generated images are more photo-realistic and contain the high-frequency details such as corners around the objects like cups, hands. Also, there are no pixelated boxes as compared to generated images from MSE trained model. Thus, we can infer that the SRGAN model trained with PERPIX loss has started learning the high-frequency details in less number of training steps with more weight towards the VGG loss component of the PERPIX loss.

<img src ="downloaded images/tensorboard/mse real.png" width = "400" height = "500" /> <img src ="downloaded images/tensorboard/mse gen.png" width = "400" height = "500" /> 

<p align="center"> Figure: Original images (on left), generated images from MSE trained model (on right) <p align="center">

<img src ="downloaded images/tensorboard/perpix real.png" width = "400" height = "500" /> <img src ="downloaded images/tensorboard/perpix gen.png" width = "400" height = "500" /> 

<p align="center"> Figure: Original images (on left), generated images from PERPIX trained model (on right) <p align="center">

To conclude, both in terms of pixel and perceptual qualities, the PERPIX trained model performs better and considered as the best model among two models implemented in this project and showed satisfactory results compared to models from Ledig et al. [2] ***(but requires further training)***. PERPIX trained SRGAN model has the potential to balance between smoothening and high-frequency details of images, which is the desirable property in real-world scenarios. 

***Benefits of PERPIX loss:***

1. Using PERPIX loss, the SRGAN model preserves the pixel quality besides improving the perceptual quality of images

2. There is no requirement of a pre-trained model for the generator network

3. Flexibility to train the model with different weights giving importance towards pixel and perceptual qualities 

### Super-Resolution of videos:

Using PERPIX trained model, I have done super-resolution inferences on faces and videos, and the link to access those is provided below.

[Super-Resolution of Videos](https://drive.google.com/drive/folders/1MZ2jsJh2iKKoOwUOTzdIGkF3u_Zen2lF?usp=sharing)

### Deployed Models:

Now, the trained models (best models in case of MSE and PERPIX losses) were deployed in the Google Cloud AI platform and can be used for future predictions from almost any application through a REST API call sending JSON payload of shape 4- Dimensional array or tensor, where the first dimension corresponds to the batch size of images and fourth dimension represents the number of color channels of images. The link to the deployment website has been provided below:

[Deployed Models](https://console.cloud.google.com/ai-platform/models?authuser=3&project=srgan-deploy)


### Project Workflow and Training Details:

I have trained the models using the Adam optimizer [4] with β1 = 0.9 and learning rate of 0.0001 for generator and discriminator respectively. For each training step, model training was alternated between the generator and discriminator that is k=1 as described in [1]. In total, I have trained both models (MSE loss and PERPIX loss trained models) for 3.5 * 10^4 steps (where each step is training over a mini-batch of training data). The figure illustrating the basic workflow of this project is provided below. To describe the workflow, both the models were trained in parallel with NVIDIA Tesla P100 GPUs, one on Google Colaboratory and the other on Google Cloud’s AI platform Deep Learning virtual machine.

During phase 1 of this project, the model was built and trained for a few training steps. In phase 2, GPU training, TensorBoard, other code implementations, and bug fixes were done. Later, the model was trained and deployed in the Google Cloud Platform. The following figure demonstrates the workflow of this project involving various tools. For Google Colab training, the data is taken from Google Drive for training. On the other hand, for Google cloud AI Deep learning VM, data was fetched from Google cloud storage. All the models were implemented using TensorFlow (python) and are compatible to run on different platforms. The trained models were then deployed on to the Google Cloud AI platform. Those deployed models have been used for performing inferences and can be used for future predictions of super- resolution on images or videos.

<img src ="downloaded images/workflow.png" width = "900" height = "500" align = "center" /> 

<p align="center"> Figure: Project Workflow <p align="center">

### Status of this project:

All of the mentioned project goals were accomplished successfully and currently working on application development part of this project, and research with deep learning techniques to improve performance is ***ongoing*** . 

### Advantages of Super-Resolution:
 
- It saves the storage space of images and provides high resolution images whenever needed

- Adapts to new hardware upgrades(like improved screen resolution of TV, Theatre, etc)

- Make objects to be highly distinguishable in images so that data as whole will be useful for other computer vision tasks through pre-processing or post-processing.

- Satellite Imagery with blurred images


## References

[1] I.Goodfellow, J.Pouget-Abadie, M.Mirza, B.Xu, D.Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, 2014.

[2] C. Ledig, L. Theis, F. Husza ́r, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017. [(SRGAN)](https://arxiv.org/abs/1609.04802)

[3] Zhihao Wang, Jian Chen, Steven C.H. Hoi, Fellow, "Deep Learning for Image Super-resolution: A Survey", IEEE, 2020.

[4] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[5] Nao Takano and Gita Alaghband. “SRGAN: Training Dataset Matters”, 2019. ( arXiv:1903.09922 ).

[6] Datasets Link:
1. DIV2K Dataset [Dataset-link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. MS-COCO Dataset [Dataset-link](http://cocodataset.org/#download)
3. Vid4 Dataset [Dataset-link](https://xinntao.github.io/open-videorestoration/rst_src/datasets_sr.html)
4. LFW:[Dataset-link](https://www.tensorflow.org/datasets/catalog/lfw)
5. Set5 and Set14: [Dataset-link](https://www.kaggle.com/ll01dm/set-5-14-super-resolution-dataset)

[7] Agustsson, Eirikur and Timofte, Radu. “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study”, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[8] C.Y. Yang, C. Ma, and M.H. Yang. Single-image super-resolution:A benchmark. In European Conference on Computer Vision (ECCV),pages 372–386. Springer, 2014.

[9] C.Y. Yang, C. Ma, and M.H. Yang. Single-image super-resolution:A benchmark. In European Conference on Computer Vision (ECCV), pages 372–386. Springer, 2014.

[10] [Tensorflow-Documentation](https://www.tensorflow.org/)
