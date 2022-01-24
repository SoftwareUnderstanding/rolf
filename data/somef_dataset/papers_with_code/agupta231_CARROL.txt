# C.A.R.R.O.L. 
**C**ontext **A**ware supe**R** Res**OL**ution

## Problem Description 
Super resolution is the problem of “adding more pixels” in an image. To elaborate upon that, super resolution is a class of techniques to generate extra pixels in an image by extrapolating data from the picture as a whole. Currently, most techniques attempt to solve this problem by mathematically analysing an area and extrapolating the trends to intelligently guess what the values of the pixels would be at a higher resolution. We are interested in exploring deeper into Generative Adversarial Networks (GANs), which is a type of solution to the super resolution problem.
GANs are the technique of using two deep neural networks to generate statistically accurate data points. In the application of super resolution, GANs treat pixels as the sample space, and try to infer the value of new pixels from that. However, most GANs currently solve the problem by just looking at pixel data, shapes, and other visual data found in the pictures. We are interested in seeing how “context” affects GANs’ performance. We want to see how adding metadata that is not necessarily apparent from the mathematical pixel data affects the GANs’ improvement in enhancing a photo, as well as GANs’ improvement in the range of photos that it can enhance.
## Motivation 
The inspiration behind this project comes from the numerous crime-solving themed movies in which investigators use “zoom and enhance” features on images and videos. However, given that super  resolution on its own has been done a number of times before, we wanted to add our own twist to it by adding an image classifier that would provide some sort of context as to what the image might be for the super resolution algorithm to make a more educated enhanced image.
## Methodology
Our project will be split into two parts. The first part will be training a classifier algorithm using tried and true methods to categorize various elements in our dataset. Then, we will use the classification label as another input in our super resolution model in hopes of creating better images. For example, imagine the difference between giving the super resolution model an image and giving the super resolution model an image and labeling it as a dog. We want to see if training our model to recognize context will affect its overall performance. We will then compare our model’s results with the results from other super resolution algorithms using either inceptions scores or Frechet Inception Distance.
## Datasets and Tools 
Our team is going to use Keras on top of either Theano or Tensorflow to build our network. We plan to use the standard datasets from industry to test and compare our network.
Urban 100 dataset: Images of Buildings
BSD 100 dataset: 
Sun-Hays 80 dataset 
 
# References
https://arxiv.org/pdf/1606.03498.pdf

https://arxiv.org/pdf/1711.10337.pdf

https://arxiv.org/pdf/1706.08500.pdf

https://sites.google.com/site/jbhuang0604/publications/struct_sr
