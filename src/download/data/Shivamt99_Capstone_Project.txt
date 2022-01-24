# Capstone_Project
Final Capstone Project

## WHO LET THE DOG'S OUT?


# Objective

This project hopes to identify dog breeds from images. This is a fine-grained classification problem: all breeds of Canis lupus familiaris share similar body features and over-all structure, so differentiating between breeds is a difficult problem. Furthermore, there is low inter-breed and high intra-breed variation; in other words, there are relatively few differences between breeds and relatively large differences within breeds, differing in size, shape, and color.

At 202 breeds recognized by the American Kennel club, even canine enthusiasts could struggle to correctly identify most breeds from images. I propose an application that can take images of a dog and gives its best estimation of the dog breed, hopefully exceeding the average human level of accuracy. I believe an application such as this will come to the aid of municipal animal control services and good Samaritans recognizing an opportunity to reunite a lost dog with their family.



# Data Info

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Contents of this dataset: 

•	Number of categories: 120

•	Number of images: 20,580

•	Annotations: Class labels, Bounding boxes

Source Link: http://vision.stanford.edu/aditya86/ImageNetDogs/main.html


# Files

Path_setup_Preprocessing.ipynb - Set directory and preprocess the data

CNN_Model - CNN architecture built from scratch

Transfer_Learning - Implementation of Vgg16 transfer learning

# References

http://cs231n.stanford.edu/

Learning Multi-attention Convolutional Neural Network for Fine-Grained Image Recognition”, Heliang Zheng, Jianlong Fu, Tao Mei, Jiebo Luo, 10.1109/ICCV.2017.557, IEEE

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. http://www.deeplearningbook.org.
J. Liu, A. Kanazawa, D. Jacobs, and P. Belhumeur, “Dog Breed Classification Using Part Localization”, Computer Vision–ECCV 2012. Springer Berlin Heidelberg, 2012. 172-185.

A. Krizhevsky, I. Sutskever, and G. Hinton. “Imagenet classification with deep convolutional neural networks”, Advances in neural information processing systems. 2012.
sakshm789

https://arxiv.org/abs/1409.1556

http://ijarcet.org/wp-content/uploads/IJARCET-VOL-5-ISSUE-12-2707-2715.pdf

http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf

https://web.stanford.edu/class/cs231a/prev_projects_2016/output%20(1).pdf
