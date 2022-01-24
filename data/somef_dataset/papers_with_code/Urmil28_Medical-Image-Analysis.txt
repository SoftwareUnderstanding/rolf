# Medical-Image-Analysis

Code for the paper "Towards designing an automated classification of Lymphoma Subtypes using Deep Neural Networks" by Rucha Tambe, Sarang Mahajan, Urmil Shah, Mohit Agrawal and Bhushan Garware to be published in the ACM India Joint International Conference on Data Science & Management of Data, 2019 (CoDS-COMAD).

This NIA curated dataset [1] helps to address the need of identifying three sub-types of lymphoma: Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), and Mantle Cell Lymphoma (MCL). Currently, to distinguish between the sub-types additional costs and equipment overheads are encountered and only then can the treatment of the disease be decided. A successful approach using Deep Learning should allow for more consistent and less demanding diagnosis of this disease.
This dataset was created to mirror real-world situations and samples prepared by different pathologists at different sites. Selected samples also contain a largeer degree of staining variation than one would normally expect. In this work we attempt to separate images into 1 of 3 sub-types of lymphoma. Whole tissue samples are fed to the DL algorithm to learn unique features of each class which are used for classification.

Approach:
We used an Inception V3 Network [2] consisting of several branches which are used to determine the appropriate type of convolution to be made at each layer. Inception V3 Network differs from a CNN in that, it uses multiple filter sizes along with convolutional layers, pooling layers in each different branch of the model. Finally, the output of the branches is concatenated. Hence, Inception V3 can be thought of as a network comprised of several cluster units, using multiple layers, the outputs of which are concatenated to form a chain of clusters, eventually enabling classification.
We use the Inception V3 Network architecture as the basis, and modified it to suit our dataset. The description of the architecture is given below.

The base Inception V3 model is used and additional layers are added as below:

![base](https://user-images.githubusercontent.com/32006882/47720927-0819e800-dc75-11e8-9ce4-875baa0f3432.png)



Before feeding training images to this deep neural network, we resize the images originally having dimensions of 1388 X 1040, to 347 X 260. We choose a factor of 4 to reduce the  image size. The total parameters as a result of this model are 21,956,687 out of which, 21,992,255 are trainable and 34,432 are not trainable.
During training of the model, we also perform online data augmentation of the dataset
During the testing phase, we gave 75 images as input and classify the image according to the prediction of the softmax layer of our model.

References:

1] Dataset: http://www.andrewjanowczyk.com/use-case-7-lymphoma-sub-type-classification/

2] Inception V3: https://arxiv.org/pdf/1512.00567v3.pdf
