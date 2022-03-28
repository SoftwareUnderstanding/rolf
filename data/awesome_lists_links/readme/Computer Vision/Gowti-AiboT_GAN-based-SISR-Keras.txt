# GAN-based-SISR---Keras

**Target:**

Generate the Photorealistic high resolution images(image size -196x196) from a single low resolution image (implemented image size-49x49).

**For detailed Information on concepts:**

https://drive.google.com/file/d/1w_C4LDSzgiqpDMArdcvPZtYz3gDNFGE_/view?usp=sharing


**GAN based SISR Architecture:**
![Image of Architecture](https://github.com/Gowti-AiboT/GAN-based-SISR-Keras/blob/master/Architecture_Images/architecture.jpg)

**Generator and Discriminator Network:**
![Image of Network](https://github.com/Gowti-AiboT/GAN-based-SISR-Keras/blob/master/Architecture_Images/network.jpg)

**Attachment Structure**

1.Network.py : Contains Generator and Discriminator Network

2.Utils.py   : Contains utilities to process images

3.Utils_model.py : Contains optimizer and content loss code

4.train.py   : Used for training the model

5.test.py    : To test the model


**Working Enivornment**

1.Training dataset – COCO2019

2.Test dataset – Set5

3.Training size – 800 images

4.Scale – 4x

5.LR Image size – 49 * 49 , HR Image size  - 196*196

6.Batch size – 16

7.GPU used – Nvidia Tesla K80(colab Allocation)

8.Platform Used : Google colab , Spyder (Local Machine-Laptop)

9.Tools Used : Keras ,Backend – Tensorflow , numpy,PIL ,os, argparse , skimage

10.Time Taken – 3 hours /100 epochs.

**Output:**

Image Result format - (LR , SR , HR)

**900epochs**

Test results are not satisfactory when trained for 900 epochs and batch size-16
![Image of 900epochs](https://github.com/Gowti-AiboT/GAN-based-SISR-Keras/blob/master/900epochs.png)

**3000epochs**

Test results are satisfactory when trained for 3000 epochs and batch size-64
![Image of 3000epochs](https://github.com/Gowti-AiboT/GAN-based-SISR-Keras/blob/master/3000epochs.png)

**Evaluation_table**

![Image of table](https://github.com/Gowti-AiboT/GAN-based-SISR-Keras/blob/master/table.png)

**Paper Reference**

https://arxiv.org/abs/1609.04802

**Reference:**

1.	https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112
2.	https://github.com/deepak112/Keras-SRGAN
3.	https://github.com/leftthomas/SRGAN
4.	https://www.slideshare.net/reachquadri/what-is-spatial-resolution
5.	https://medium.com/@jonathan_hui/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
6.	https://www.cambridgeincolour.com/tutorials/image-interpolation.htm
7.	https://medium.com/beyondminds/an-introduction-to-super-resolution-using-deep-learning-f60aff9a499d
8.	https://arxiv.org/abs/1609.04802
9.	http://www.ee.iisc.ac.in/people/faculty/soma.biswas/AIP_pdf/SBiswas_ImageSuperresolution_2016.pdf 
10.	https://download.atlantis-press.com/article/4822.pdf
11.	http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/Super_Resolution_CVonline.pdf
12.	http://people.csail.mit.edu/billf/publications/Example-Based_Super_Resolution.pdf
13.	https://www.researchgate.net/publication/260737170_Performance_evaluation_of_image_quality_metrics_with_respect_to_their_use_for_super-resolution_enhancement
14.	https://arxiv.org/abs/1501.00092
15.	https://slideplayer.com/slide/4838896/
16.	https://arxiv.org/pdf/1502.03167.pdf
17.	https://arxiv.org/abs/1809.00219
18.	https://ieeexplore.ieee.org/document/8517442









