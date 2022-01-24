# CoMBInE
*Cosmic Microwave Inpainting Experiment*

CoMBInE is a machine learning program that through inpainting aims to reconstruct patches of the Cosmic Microwave Background thermals maps, as new nowel approach to eliminate/limit the effect of the foreground.

In this program we used the innovative method developed by a group of researches from NVIDIA corporation which uses Partial convolutional layers in place of the traditional convolutional layers. Traditional CNN filter responses are conditioned on both valid pixels as well as the substitute values in the masked holes which may lead to color discrepancy and blurriness. The partial convolutions are constructed such that, given a binary mask, the results depend only on the non-hole regions at every layer. Given sufficient layers of successive updates and with the addition of the automatic mask update step, which removes any masking where the partial convolution was able to operate on an unmasked value, even the largest masked holes will eventually shrink away, leaving only valid responses in the feature map.
See paper for more details: "*Image Inpainting for Irregular Holes Using Partial Convolutions*", https://arxiv.org/abs/1804.07723. 


# Dependencies
* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12

# Implementation details
Details of the implementation are in the [paper itself](https://arxiv.org/abs/1804.07723), here a short summary of the main features.

## Mask Creation
The masks we generate are of two kinds and they both used OpenCV to draw their shapes. The first method contained in the script *utils/Mask_generator.py* creates masks composed of ellipses, lines and circle of random size and position. The second method contained in the script *utils/Mask_generator_circle.py* creates circular masks with center and radius that can vary according to one's preference. For simplicity, in the analysis done so far, the circular masks were always centered at the center of the images.

## Partial Convolution Layer
The key element in this implementation is the partial convolutional layer. Basically, given the convolutional filter **W** and the corresponding bias *b*, the following partial convolution is applied instead of a normal convolution:

<img src='./images_and_plots/read_me_equations/eq1.png' />

where ⊙ is element-wise multiplication and **M** is a binary mask of 0s and 1s. Importantly, after each partial convolution, the mask is also updated, so that if the convolution was able to condition its output on at least one valid input, then the mask is removed at that location, i.e.

<img src='./images_and_plots/read_me_equations/eq2.png' />

After each convolution, the mask is removed at the location where convolution was able to condition its output on the valid input. The result of this is that with a sufficiently deep network, the mask will eventually disappear (be all ones).

## UNet Architecture
The architecture is UNet like, replacing all convolutional layers with partial convolutional layers and using nearest neighbor up-sampling in the decoding stage. The skip links will concatenate two feature maps and two masks respectively, acting as the feature and mask inputs for the next partial convolution layer. The last layer’s input contains the concatenation of the original input image and original mask to allow model to copy non-hole pixels (in this way the image in the non-hole pixels should basically corresponds to the original/target image). 

## Loss Function(s)
The loss function used in the paper is very convoluted and to review it, it is preferable to refer to the paper. In short it includes:

* Per-pixel losses both for maskes and un-masked regions
* Perceptual loss based on ImageNet pre-trained VGG-16 (*pool1, pool2 and pool3 layers*)
* Style loss on VGG-16 features both for predicted image and for computed image (non-hole pixel set to ground truth)
* Total variation loss for a 1-pixel dilation of the hole region

The weighting of all these loss terms are as follows:
<img src='./images_and_plots/read_me_equations/eq7.png' />

## Training Procedure
CoMBInE was trained on 1000 simulated CMB thermal maps created with CAMB. From each map it was taken only the central area, dividing it in 10 square patches 128x128 pixel each. In total the program was thus trained on 10000 images.
The dataset was divided in Training, Validation and Test set, corresponding respectively to 70%, 15%, 15% of the total dataset. Training was furthermore performed using the Adam optimizer in two stages since batch normalization presents an issue for the masked convolutions (as mean and variance is calculated for hole pixels).

**Stage 0** (*combine_initial_weight_computation.py*)
Train the program using only 1 image as both training, test and validation set for 10 epochs with 2000 steps each. The goal of this state is to obtain the weights that will be used to initialize stage 1.

**Stage 1**
Learning rate of 0.0001 for *n1* epochs with batch normalization enabled in all layers

**Stage 2**
Learning rate of 0.00005 for *n2* epochs where batch normalization in all encoding layers is disabled.

The number of epochs *n1*; *n2* to run in each step should be determined based on the evolution of the validation loss and total loss. If they stop steadily decreasing, move on to the next step.


# How to use this repository

## Libraries
The libraries were essentialy taken from the Git repository https://github.com/MathiasGruber/PConv-Keras where Mathias Gruber tries to reproduce the results obtained in the paper "*Image Inpainting for Irregular Holes Using Partial Convolutions*". They were then modified in order to account for the needs of this research.

*Mask Generator (Random shapes)*

Generates masks made of ellipses, circles and lines in random order, amount and size to cover about 25% of the input 128x128 pixel image.

*Mask Generator 2 (Circle with various radius)*

Generates circular mask of various radius and center position that can cover from 0 to 100% of the input 128x128 pixel image. This masking is very useful to measure the ability of CoMBInE to recontruct the image based on the percentage of area covered.

*PConv UNet Architecture (512x512): utils/PConv_Unet_model.py*

This model takes in 512x512 pixel images. The coding and decoding stages of the UNet architecture are made of 8 layers. In the coding phase the number of filters in each convolution steadily increases from 64 to 512 meanwhile the kernel size decreases from 7 to 3. In the decoding stage, the kernel size is kept constant at 3 meanwhile the number of filters in each convolution decreases steadily until the final concatenation with the input image where it is equal to 3.

*PConv UNet Architecture (128x128): utils/PConv_Unet_model_100.py*

This model takes in 128x128 pixel images. It was mainly constructed for purposes of time optimization, since with 512x512 pixel images the training was very slow an it is what was used in this research. The coding and decoding stages follow the same structure in terms of filters and kernel sizes but the number of layers is reduced to 7.

## Training


**Old Dataset:** *combine_training.py*

The batch size was chosen to be 3 and each epoch was composed of 10000 steps including 1000 validation steps. In total CoMBInE was trained for 42 epochs, respectively 25 epochs for the first phase and 17 for the second phase. The learning rate was very steep during the first 20 epochs and then levelled off with the validation loss that specifically stopped to monotonically decrease. For more details see picture */images_and_plots/CMB_inpainting_percent_plot2_phase1_trainset_10000-2.jpg*

**New Dataset:** *combine_training_new.py (work in progress)*

The images are divided in 10 classes corresponding to the 10 different values of the scalar spectral index ( ***SSI***) used. All spectral indeces were chosen between 0.5 and 1.5. More specifically the values of ***SSI*** are [0.5;0.7;0.9;0.95;1;1.05;1.1;1.3;1.5]. The range is more dense around 0.95 which corresponds to the value observationally better contrained. Indeed, this choice was made in order to achieve better results for CMB maps with parameters closer to the one observationally measured (by better results we mean enhancing the ability of CoMBInE, for ***SSI*** near 0.95, to distinguish maps in regard to the ***SSI*** they belong to).
The batch size was increased to 16 and the number of steps per epoch and the validation steps were changed accordingly to that and the total new number of images in the dataset (specifically 6250 steps and 625 validation steps). The motivation for this increase in the batch size is to ensure that CoMBInE passes minimum through one image per class before updating its weights. That said the batch size is still not definite, the program has not yet finished its training so the total number of epochs is still unknowned and some of the parameters may be tuned in the meantime.

**Loading the model and chunked training process**

Recall that, given the structure of CoMBInE, there is no need to worry about determining a total number of epochs to run for the training. Indeed we can simply run it for a small number of epochs and then based on the improvement in the total and validation loss, restart the training uploading the last updated weights and follow the same procedure.

**Important note on Pre-trained weights**

The model must be always at first initialized with the VGG 16 weights from ImageNet before loading the new weights obtained after the training and use them to make valid predictions. These pre-trained VGG16 weights are only useful to us in order to classify our images in terms of their respective value of ***SSI***. The VGG16 model for Keras is the Keras model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition which has the capability of classifying objects in photographs. The model was trained on ImageNet which is a huge database of millions of images, thus there is no point in creating our own model or retraining the VGG16 model on our dataset which is much simpler than that of ImageNet. Luckily we can just used their pre-trained weights which are available for free and can be found at this link:
* [Ported VGG 16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0)

For more information on the VGG16 model refer to this paper: https://arxiv.org/abs/1409.1556

## Analysis: Target vs Prediction comparison 

To determine the efficency of CoMBInE in reconstructing the CMB map square chunks we established a set of parameters that could help us both analitically and visually understanding the performance of the program.

**Mean and Standard Deviation Gross Difference:**

The first parameter defined is the "gross" mean of the difference between predicted and target image. By gross mean we refer to the total mean of the image of the absolute value of the difference taken within the 3 values of R,G and B color for all pixels in the image. Theoretically, for a perfect recontruction, this value should be always zero. The second parameter defined, which relates to this mean difference, is its standard deviation of all the pixels from the image of the absolute difference. Again, for a perfect reconstruction, this value also tends to zero. To get a more detailed idea wheter or not the mean difference computed is actually small or not, we also defined the mean percentage difference where the percentage is calculated with  respect to the target. 
The analysis python scripts that refer to these parameters are *target_vs_prediction_random-images_gross_difference.py* and *target_vs_prediction_vs_mask-size_gross_difference.py*. In the first one we analyze the predictions after each epoch for 50 random images taken from the  validation set masked with 1 mask made of random shapes. In the second one we analyze the predictions after each epoch for 1 image, given possible masking sizes and shapes (specifically we use 1 random shape mask and 12 centered circular masks with radius ranging from 5 to 65).

**RGB and Grey scale Color Distribution**

In this analysis the distribution for each color (red, green and blue) in the predicted image is individually plotted and compared with the target distributions. The parameters used for the comparison are the mean and standard deviation of the Gaussian fitted to each distribution. This same analysis is then repeated for the grey scale which is simply defined as:

<img src='./images_and_plots/read_me_equations/grey_scale.png' />

where **R**, **G**, **B** respectively represent the red, green and blue color distribution. In *target_vs_prediction_color-distribution.py* we analyze the predictions after each epoch for 1 image, randomly chosen from the validation set masked with 1 mask of random shape. 

**Final epoch image outputs**

This section refers to the last analysis script *Update target_vs_prediction_last_epoch_images_and_distribution.py* which simply compares the perdiction at the last epoch with a target image, randomly chosen from the validation set, in term of the color distribution and also serves to visualize such comparison by plotting side by side and in this order target, predicted and absolute difference image. Below an example of our results taken from the *images_and_plots/* folder where you can find also all the other plots and figures we have produced this far.

<img src='./images_and_plots/plot_target_pred_diff.jpg' />

**NOTE:** this analysis method only refers to the **old** dataset. The scripts for the new datasets have not been made yet but they will simply result in small modifications of the current scripts to account for the larger dataset and the differentiation in classes based on the value of **SSI**.

## Future Work

**New Dataset Training and Analysis**

In regards to the training part, it needs to be verified wheter or not changing batch size could optimize the learning of CoMBInE. After the conclusion of the training, the analysis, as described above, will be repeated also comparing wheter or not there are non neglegible differences in the learning and the predictions betweeen images from different classes

**Computing Power Spectra**

The problematic with computing power spectra from RGB images are several. First of all there is a dimensional problem since each pixel is composed by a 3 dimensional array corresponding to the values of red, green and blue. Thus we need to reduce the RGB images to Grey scale. This is rather simple but unfortunately does not fix the full problem. Indeed any image repsentation of the CMB map (like the ones simulated by CAMB) do not encode all the information about the temperatures at each point in the sky. Indeed when one simulates CMB maps for visualization purposes, a limit is set on the range of temperatures such that *T* can only vary within *T_min* and *T_max*. This implies that each point at a higher temperature than *T_max* is assigned the same color as those at excactly *T_max*. The effect on the power spectrum should not be that significant but it's definetely something to take into account as it would schew the final results and eventually **the goal of CoMBInE is to compute power spectra from actual CMB data so we need to be precise**. The strategy is then twofold, on one side we can compare these schewed power spectra of the predicted and target image as another way to test to what extent CoMBInE is able to reproduce CMB maps. On the other side we need to change the architecture of the program such that it can intake 2 dimensional arrays where to each pixel would correspond 1 number proportional to the temperature and not limited to the any range. This requires some work as all the convolutional layers were build to intake images with dimensions (128x128x3) where the filter matrices had specific kernel dimensions which change accordingly based on the initial image size.

**Validate our results with real data from the CMB temperature map**

This would be the final stage of the project as eventually we would like to use CoMBInE to fill in the holes of the patches of the sky where the effect of the foreground is not neglegible, to then compute a Power Spectrum of the entire CMB map.

**optimize, optimize, optimize!**

Whatever results we obtain we could probably do better by changing some parameters of or some parts of the architecture
