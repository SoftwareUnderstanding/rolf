<br>

# Image Restoration using Pix2Pix

<p align="center">
  <img src="/ImagesInText/005158_real_A.png" width="100%" height="100%"><br>
  Figure 1: From left to right: Pixelated image - BS64 N10 - BS64 N40 - BS64 N40 LRx20 - BS1 N10 - Real image
</p>


## Acknowledgements
We would like to thank our teaching assistant Pranjal Rajput for his guidance and advice during this project. He helped us think more outside of the box and steered the project in the right direction. Furthermore, we would also like to give thanks to Jan van Gemert for his enthusiasm while giving lectures and motivating us to successfully complete this project.

## Introduction
The paper Image-to-Image Translation with Conditional Adversarial Networks (https://arxiv.org/pdf/1611.07004.pdf) showed that a general purpose solution could be made for image-to-image translation. Since the time that this paper was published, multiple artists and researchers have made their own models and experiments, with stunning results. These models range from creating cats out of drawings to creating videos of the sea by using an input from general household appliances.<br>
The objective of the pix2pix model is to find a model that can map one picture to a desired paired image, which is indistinguishable from the real thing. An example is shown in Figure 1, where 4 different models attempt the mapping from the pixelated image to the real image. Pix2pix uses Conditional Generative Adversarial Networks to achieve this objective. Conditional means that the loss here is structured, there exists a conditional dependency between the pixels, meaning that the loss of one pixel is influenced by the loss of another. The loss function that is used by the model is shown in Equation 1.

<p align="center">
  <img src="/ImagesInText/LossFunction.png" width="60%" height="60%"><br>
  Equation 1 [P. Isola et al. (https://arxiv.org/pdf/1611.07004.pdf)]
</p>

The model can then be trained and results evaluated, which has been done for a great variety of experiments. We wanted to see if another application could be made, that of image restoration of blurry pictures. We have most likely all seen a movie or tv-series in which a spy agency needed someone to "enhance" a photo in order to see smaller details, with the advent of deep learning these techniques are becoming more of a reality. We wanted to see if this general architecture of pix2pix for image translation could also be used for this application to see if you could enhance your images by using pix2pix.

This blog first starts with the method of our project, what exactly is the type of data that we investigate and in what type of datasets they are stored. This is followed up by an explanation about the hyperparameter tuning that was performed and why this was important. The last part of the method is how we would evaluate our results. The method is followed up by our experiments, which also gives a sample of the data that we used as well as the result of some of our experiments, these results are then discussed in our discussion as well with our conclusion about the experiment if pix2pix can be used for image restoration.

## Method
In this section, four topics will be touched. Firstly, a short description of the datasets is given and will be further elaborated upon later in the text. Secondly, insight is given in how the models were trained and tested. Thirdly, the hyperparameters that could be changed are discussed. Fourthly, it is explained how the performance of the models is evaluated.

### Datasets
For our initial research it was decided to use images of which the resolution was lower than that of the target/true image. Pix2pix generally uses 256x256 images, which is the size that we want to create. For input we wanted to initially use a dataset which used 64x64 images, but later an additional dataset was created where the resolution of the pictures was allowed to vary between 48x48 to 128x128 pixels, because initial tests showed that the model did not generalize well on random photos taken from the internet.

### Algorithm Training
To train the algorithm a github repository was set up, in which the original pix2pix repository was cloned (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Later we included our own data in this repository. This repository could be cloned to the virtual machine provided by Google Colaboratory, using the following code:
```python
!git clone https://github.com/PieterBijl/Group28.git/
```
From here the working directory should be set to the cloned folder and the requirements for pix2pix should be installed. This was done using the following piece of code:
```python
import os
os.chdir('Group28/')
!pip install -r requirements.txt
```
Or alternatively with:
```python
cd /content/Group28/
!pip install -r requirements.txt
```
The following snippet of code is an example for training the algorithm with batch size 64, 50 normal epochs and 0 decay epochs:
```python
!python train.py  - dataroot ./datasets/variational_data  - name variational_data_batch_size_64_normal_epochs_50_decay_0  - batch_size 64  - model pix2pix  - n_epochs 50  - n_epochs_decay 0
```
The `./datasets/variational_data` refers to the data stored in the folder `datasets/variatonal_data`. In this folder two folders are located: `train` and `test`. As the name suggests, the training data is located in the train data folder and the test data in the test data folder.
Similar to `-n_epochs`, as used above, the optional parameters for training are the following:
- n_epochs
- n_epochs_decay
- beta_1
- lr
- gan_mode
- pool_size
- lr_policy
- lr_decay_iters

After discussing the location and the testing of the results, the optional parameters will be discussed more in detail in the following section. The resulting models are stored in the folder checkpoints and can be accessed using the following command:
```python
Cd /content/Group28/checkpoints/variational_data_batch_size_64_normal_epochs_50_decay_0/
```
This data can be stored for later use and can be tested using the following piece of code:
```python
!python test.py  - dataroot/content/Group28/datasets/variatonal_data/test/  - name variational_data_batch_size_64_normal_epochs_50_decay_ 0  - model test
```
These results were used for later analysis of the performance of the model.

### Changing hyperparameters

In their code there are several possible candidates for hyperparameter tuning. The possible candidates were:
- Number of epochs with a constant learning rate, default = 100.
- Number of epochs with a decaying learning rate, default = 100.
- Momentum term of Adam optimization algorithm, default = 0.5.
- The initial learning rate for the Adam optimization algorithm, default = 0.0002.
- The GAN mode, of which the options are vanilla, lsgan and wgangp, default = lsgan.
- The pool size, which is the size of the image buffer that stores previously generated images, default = 50.
- The learning rate policy, which can be linear, step, plateau or cosine, default = linear.
- The learning rate decay iterations which multiplies the learning rate by a gamma every lr_decay_iters iterations, default = 50.

Additionally we looked at the algorithms training performance for different batch sizes. Using a batch size greater than one and smaller than the training set size leads to mini-batch gradient descent, in which the internal model parameters are updated after a number of samples have been worked through. This mini-batch training both requires less memory and trains faster compared to both stochastic gradient descent, where the batch size is equal to 1, and batch gradient descent, where the batch size is equal to the training set size.

### Performance Measurements
In their paper, P. Isola et al. use a perceptual study on the generated images. Turkers from Amazon Mechanical Turk were shown 40 images and for each they had to tell whether it was a generated image or an actual image. Since, for our study, multiple hyperparameters were changed, there was a large number of models created. A perceptual study on the generated images would simply take too much time, so instead a quantitative method seemed more interesting. Such a method is also used in the paper: FCN score. If a semantic classifier is trained on real images and the generated image looks real, it should also be able to classify the generated image successfully. However, a dataset would then have to be created which provides the true semantic segmentation of the images, which would again be too time consuming. Instead, it was decided to go for another method that is able to quantify the similarity between the two images: The Peak Signal-to-Noise-Ratio, from now on called PSNR.

The PSNR is a measure of similarity between the generated images and the original images. First it calculates the mean squared error, from now on called MSE, of the generated image, which can be seen in Equation 2. It subtracts the pixel values of the generated image from the real image and then sums the square of these outcomes. In the last step, this is divided by the amount of pixels in the image, 256x256.


<p align="center">
  <img src="/ImagesInText/MSE.png" width="60%" height="60%"><br>
  Equation 2 [Mathworks (https://nl.mathworks.com/help/vision/ref/psnr.html)]
</p>

The calculation of the PSNR is shown in Equation 3, where R is the maximum pixel value, 255.

<p align="center">
  <img src="/ImagesInText/PSNR.png" width="60%" height="60%"><br>
  Equation 3 [Mathworks (https://nl.mathworks.com/help/vision/ref/psnr.html)]
</p>

The higher the MSE, the lower the PSNR will be. So, when testing the models, higher values for PSNR will result in better performance. Furthermore, as a double check for the results, the cosine similarity was also computed using Equation 4. Every single pixel of the generated image is multiplied with the pixel at the same coordinates in the real image. Then, this product is divided by the multiplication of the magnitudes of both images. The magnitude of an image basically means a dot product by itself.

<p align="center">
  <img src="/ImagesInText/Similarity.png" width="60%" height="60%"><br>
  Equation 4 [Neo4j (https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/)]
</p>

To conclude, every model was tested for performance and a value was given to its PSNR and its cosine similarity. After testing all the models, the combinations of these scores were plotted in Figure 2. A trend line can be seen which shows the relationship between the two values. When the model gets a high value for its PSNR, it will also get a high value for its cosine similarity, and vice versa. The code can be found at Evaluation.py.

<p align="center">
  <img src="/ImagesInText/Figure2.png" width="70%" height="70%"><br>
  Figure 2: Cosine similarity scores plotted against PSNR scores
</p>

## Experiments
This section will inform about how the datasets are created and it will discuss the selected hyperparameters to tune.

### Creating the datasets
As specified in the method section, two datasets were created: one with images with a resolution of 64x64 pixels, scaled up to 256x256 and one where resolution was allowed to vary between 48x48 pixels to 128x128 pixels, also scaled up to 256x256. To create these datasets the van gogh dataset from pix2pix was used, this dataset included over 6000 images varying from landscapes to persons, all with a resolution of 256x256. Using python these images were used to create the two datasets that were used as the input. 
```python
from PIL import Image
import os
import numpy as np
#%% Functions
def datagenerator(old_directory,new_directory):
  for data_file in os.listdir(data_directory):
    #open every image from the old file
    img = Image.open(old_directory+'\\'+data_file)
    pixel_number = round(256/np.random.randint(2,4)) 
    #determine what the resolution is going to be, from 48x48 to 128x128
    imgSmall=img.resize((pixel_number,pixel_number),resample=Image.BILINEAR) 
    #reduce the image to the desired resolution
    # Scale back up using NEAREST to original size
    result = imgSmall.resize(img.size,Image.NEAREST)
    result.save(new_directory+'\\'+data_file)
  return
```
The complete dataset consisted of a set A which had all the pixelated images and a set B which contained the 'true' images. Using one of the python files provided by pix2pix these could then be combined to form a datasets AB which combined these two datasets that could be read by the program.
```python
python datasets/combine_A_and_B.py  - fold_A /path/to/data/A  - fold_B /path/to/data/B  - fold_AB /path/to/data
```
This dataset was then divided between a training set of 5000 images and a test set of 1000 images.

### Selected Hyperparameters
Next to the batch size, we choose 3 hyperparameters to tune: the number of epochs, both normal and decaying epochs, and the learning rate. First, considering the batch size, the algorithm was trained multiple times under default parameters but with increasing batch sizes. The batch size was increased with a factor of 2 each time. We found that a batch size of 64 gave the best results regarding training time on Google Colaboratory. The results in Figure 3 show the total training duration of one normal epoch and one decaying epoch. The quality difference of the output was not assessed for all these different batch sizes since it was assumed to be constant.

<p align="center">
  <img src="/ImagesInText/Figure3.png" width="70%" height="70%"><br>
  Figure 3: The runtime plotted against different batch sizes.
</p>

With respect to the epochs, the models were trained with different numbers of normal and decaying epochs, ranging from 10 normal epochs to 150 normal epochs with step size 10. These were combined with decaying epochs ranging from 0 to 100 decaying epochs, also with step size 10. Finally, the learning rate has been tuned to 0.1, 10, 15 and 20 times the default value of 0.0002.

## Results
In this section, the results of the models are presented. First, an analytical study will show the flow through finding the best model. After, a perceptual study will show the images that were generated by 4 different models and compare them to the pixelated and the real images.

### Analytical Studies
The first models were trained on dataset AB with batch size of 100. This was prior to the knowledge that a batch size of 64 was the faster training option. The model name consists of a BS, N and D: Batch Size, Normal epochs and Decay epochs respectively. The results are shown in Figure 4.

<p align="center">
  <img src="/ImagesInText/Figure4.png" width="70%" height="70%"><br>
  Figure 4: The PSNR plotted for different models for dataset AB and batch size 64
</p>

Two interesting conclusions can already be drawn from these models. The first is that the PSNR of the pixelated image was higher than the best model so far could produce and the second is that after a certain point the model does not improve anymore. In the figure this point is at 40 normal epochs and 40 decay epochs.<br>
These results were discussed with the teaching assistant and this is when the decision was made to continue with a batch size of 64. Figure 5 combines this batch size with normal epochs up to 120, for both the AB and variational dataset. Again, the PSNR of the pixelated is not achieved and the curve flattens after around 30 or 40 epochs. The models showed similar results for both datasets. The average PSNR is higher for the variational dataset, however the pixelated image also has a higher PSNR. The two black dots, originally blue, are models that are used later for comparison in a perceptual study of the images.

<p align="center">
  <img src="/ImagesInText/Figure5.png" width="70%" height="70%"><br>
  Figure 5: The PSNR plotted for different models for both datasets
</p>

It was questioned whether the flattening of the curve around 30 and 40 epochs could be prevented by using decaying epochs. Dataset AB was used in Figure 6. The figure on the left shows 30 normal epochs followed by 80 decay epochs and the figure on the right shows 40 normal epochs followed by 80 decay epochs. The figure on the right misses one data point due to an error that occurred in saving the model.

<p align="center">
  <img src="/ImagesInText/Figure6.png" width="100%" height="100%"><br>
  Figure 6 left: The PSNR plotted for different models with batch size 64. Figure 6 right: The PSNR plotted for different models with batch size 64
</p>

The figure on the left shows no major improvement in the results after 30 normal epochs. The PSNR slightly increases during the first 40 decay epochs, but it is enough to challenge the PSNR of the pixelated images. The figure on the right actually shows a decrease in PSNR after the training with decay epochs started. It also shows a small jump for 40 normal epochs. When comparing the exact same model in Figure 4, it can be seen that the first PSNR is 21.12 and the second 21.56. This caused interested in how much this difference could be for different models with the same hyperparameters. Figure 7 shows the results on the variational datasets, tested on 4 models with exactly the same hyperparameters.

<p align="center">
  <img src="/ImagesInText/Figure7.png" width="70%" height="70%"><br>
  Figure 7: The PSNR plotted for 4 identical models
</p>

Three of the results are around 22.3 and one around 22.8, meaning that the results are not always identical, but still close to each other. This deviation will not show major impact on whether the PSNR of the pixelated images can be reached or not.<br>
Since the decay epochs were not showing great impact on the results, the learning rate was changed instead. The default setting was at LR = 0.0002. Figure 8 shows the results for a learning rate that is 0.1 and 10 times the default value of 0.0002.

<p align="center">
  <img src="/ImagesInText/Figure8.png" width="70%" height="70%"><br>
  Figure 8: The PSNR plotted for models with different learning rates
</p>

The higher learning rate shows a better performance on the test data. Furthermore, even higher learning rates of 20 times the default value were tested and these showed similar results to the 10 times the original learning rate. To extend on this study, a large model with 120 normal epochs was trained on the AB dataset with two different learning rates: the default value and 20 times this value, as shown in Figure 9. Again, the black dot, originally red, is a model that is used later for comparison in a perceptual study of the images.

<p align="center">
  <img src="/ImagesInText/Figure9.png" width="70%" height="70%"><br>
  Figure 9: The PSNR plotted for models with different learning rates
</p>

Results show that the model with the higher learning rate gets a better start, but becomes unstable after 40 epochs and the PSNR of the pixelated image still remains untouched. Nevertheless, a higher maximum value of the PSNR is achieved using a higher learning rate.<br>
Still, there was no clear sign of a combination of hyperparameters that would lead to a model that could beat the PSNR of the pixelated image. It was decided to look back on the assumptions made, especially the one where it was assumed that the batch size only had an influence on training time and not so much on the result. To put this to the test, the model that takes the longest to complete one epoch, a model with batch size 1, was tested and the results are shown in Figure 10. Again, the black dot, originally blue, is a model that is used later for comparison in a perceptual study of the images.

<p align="center">
  <img src="/ImagesInText/Figure10.png" width="70%" height="70%"><br>
  Figure 10: The PSNR plotted for different normal epochs with batch size 1
</p>

These results show that the  assumption that was made can be thrown off the table. The model is almost able to reach the PSNR of the pixelated image at only 10 normal epochs. <br>
The question that remains is: How to create the most optimal model out of the results so far? It was found before that a higher learning rate showed better results, so it is worth to try this on the model with batch size 1. A model with a learning rate of 20 times the default, using 20 normal epochs and a batch size of 1 was created to put the PSNR of the pixelated images to the test. The results are shown in Figure 11.

<p align="center">
  <img src="/ImagesInText/figure11.png" width="70%" height="70%"><br>
  Figure 11: The PSNR plotted for different learning rates with batch size 1
</p>

Unfortunately, the models with a higher learning rate do not show any improvement compared to the regular learning rate and the model with the default learning rate and batch size 1 remains the model with the best performance. Now, would the PSNR values increase if the model is tested on a set of training images instead of test images? An expected answer is yes, since this is what the model's weights are trained on. To put this to the test, a model trained on the variational data with batch size 64 has been tested on both the training and the test set with different amounts of normal epochs. The results are shown in Figure 12.

<p align="center">
  <img src="/ImagesInText/Figure13.png" width="70%" height="70%"><br>
  Figure 12: The PSNR plotted for both the training and test set of the variational data with batch size 64
</p>


### Perceptual Study
In this section, the models labeled with the  black dots will be used for comparison. Ten images from the AB dataset have been selected from the test set and are used to compare four different models, as shown in Figure 13. The first column shows the pixelated image and the last column shows the expected output of the model, the real image. In between, there are results from four different models. The order of the models is based on their average PSNR on the test set, from low to high.<br>
The first model used a batch size of 64 and has been trained for 10 normal epochs. It can be seen that the model still has got difficulty in getting the colour right, plus there are quite some noisy parts in the image. Its average PSNR is 18.36.<br>
The second model used a batch size of 64 and has been trained for 40 normal epochs. The improvement is noticeable. The colours look more similar to the real image than before and in most images the noise is decreased. Its average PSNR is 21.56.<br>
The third model used a batch size of 64 and has been trained for 40 normal epochs, but the difference here is that the learning rate is multiplied by 20. The results looks like a blurry variant of the real image, but shows a good improvement compared to the pixelated image. The expected reason that this model is not the best, is because it tends to create small black holes in the image, which are disastrous for the MSE and thus the PSNR. Its average PSNR is 22.10.<br>
The fourth model used a batch size of 1 and has been trained for 10 normal epochs. Although some noise is visible, the images are sharper than the third model and no major wrong predictions are observed. Its average PSNR is 23.16.

<p align="center">
  <img src="/ImagesInText/005509_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005414_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005241_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005200_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005176_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005158_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005147_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005110_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005097_real_A.png" width="100%" height="100%"><br>
  <img src="/ImagesInText/005077_real_A.png" width="100%" height="100%"><br>
  Figure 13: For 10 test images, from left to right: Pixelated image - BS64 N10 - BS64 N40 - BS64 N40 LRx20 - BS1 N10 - Real image
</p>

## Discussion
In this section, the type of image and the evaluation method will be discussed

### Image Type
The initial results of training these models showed some promise. The model was able to generally improve the shape of objects in the pictures and fill them in with some detail. It became however quickly apparent that the model had some issues when there was significant detail in the original pictures. This is likely due to the fact that when the input images were made a lot of these details disappeared and the model is unable to generate accurate detail with this amount of training data and model complexity. That said, the model did overall quite well when dealing with landscapes and other input pictures where large surfaces were present.

### PSNR
Firstly, whether or not the PSNR is a good method to quantify the performance remains a point of interest. It seems trustworthy since it literally calculates the difference between the real image and the generated image. However, if a part of the images shows a very wrong prediction, this will be penalised heavily because the error is squared. This can for example be seen in the images belonging to the BS64 N40 LR004 model: Overall the generated images create a visually attractive result, but a certain black circle seems to appear in multiple images, causing a large MSE and thus a lower PSNR. The PSNR helps in providing an insight into whether the picture will look acceptable or not, but a perceptual study is still needed to decide which model generates the best result. <br>
Secondly, the question that came up in the research: Why is the PSNR of the variational dataset higher than the PSNR of the AB dataset? Both datasets contain the same images, however in the AB dataset the images are always 64x64 pixels and in the variational dataset they vary between 48x48 and 128x128 pixels. The average of the latter is 88x88 pixels, meaning that these images will contain more pixels, meaning more details and will thus have a higher PSNR.<br>
Thirdly, why does Figure 12 show that the PSNR values for the test images are higher than for the training images? Although the answer is not waterproof, it is suspected that the images that are included in the test set have less detail on average. The training set forms a more difficult set of images and thus results in a lower PSNR.<br>
Fourthly, why is none of the models achieving the PSNR of the pixelated images? Although it is a fair question when looking at the graphs, in reality the question is a bit too harsh. There are plenty of examples where the model generates an image that has a higher PSNR than the pixelated image. However, when taking the average over the whole test set, it turns out to be lower. As discussed before, this can be caused by a small amount of very bad predictions which pull the average down drastically. Nonetheless, bad predictions are not the only cause of the low values for the PSNR. Another explanation could be that the pix2pix model is simply not able to generate highly realistic images out of pixelated images. Its task is to turn an image of 64x64 pixels to 256x256 pixels and it uses noise to fill up the unknown pixel values. This means that 1 pixel in the pixelated image needs to be converted to 16 sharp pixels, which is a difficult job. A third explanation could be that the hyperparameters still need better tuning or that other hyperparameters should be tuned to obtain a better result.


## Conclusion and Recommendations
During this project the pix2pix algorithm was used to train models on a variety of pixelated images. The training data was pixelated with different amounts of pixels to introduce variety. The models have been trained using a wide range of hyperparameters and assessed the quality of all the different models by comparing the PSNR of test images.<br>
To conclude we find that although the pix2pix algorithm has some potential towards enhancing or restoring images, it is lacking when small details are important. Some important lessons that we have learned from this experiment are:
- Data should consist of images with a variational blur to ensure that the model can handle different types of pixelated images.
- Careful hyperparameter tuning is of great importance for optimal results.
- The PSNR and cosine similarity are not a universal panacea with respect to performance analysis.

We recommend for future work to look at the performance when training on a specific subject of images, such as landscapes only. Furthermore we recommend to perform n-fold cross validation to reduce the variance in the assessment. For the assessment we recommend to include more perceptual studies as well, since higher PSNR values don't necessarily equal better results.

If the reader is interested in other reproductions of this paper, or reproductions of other papers, we would like to to refer you to https://reproducedpapers.org/.

## Citation of the original code
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
