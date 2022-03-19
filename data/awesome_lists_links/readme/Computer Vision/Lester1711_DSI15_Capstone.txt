# DSI15_Capstone

## Product Image Classifier and Recommendation System

## Problem Statement
E-commerce sites have thousands of listings everyday and at times, the users may not be correctly classifying their uploaded image or be using the wrong product depiction. Mismatch of product listing information will decrease the effectiveness of succssful transactions and also result in unnecessary resources being utilitzed to perform these corrections on a large scale. A product detection system would help to ensure the correct listing and categorization of products or to assist the user in classifying product types. To meet this need, an image classifer will be trained and developed to accurately identify the correct image labels with the use of convulated neural networks. Customized CNN architecture and fine-tuned VGG-16 models will be used to analyse classification effectiveness. The aim is to have a self-constructed network model within 5 percentage points of a tested award winning pre-trained model.

Also, in order to boost listings and sales, it would be desirable to have an image recommendation system, based on deep learning,to present product images which are visually similar to previous posting images viewed by the user. Making use of nearest neighbours, a model will also be developed to match closest resembling products images within a particular category.


## Executive Summary

This capstone project would be aimed at building an image classification model using convulated neural networks that will identify the following 4 categories of products: backpacks, clothing, footwear and watches. The source data to construct this model will be based on 20,000 images scraped from Amazon, the world's largest online retailer. Based on this, a recognition model will be used to correctly class a certain product image. Secondly, a recommendation system would be built to promote the closest matches based on a select product image.

Stakeholders will be the e-commerce companies and the user of the services themselves. It will help the company improve the effectiveness of potential transactions. It will also improve the user experience with more accuracy and also to avoid problems arising from wrongly identifying products. Often, users may want to source for similar looking items and a recommender would help to efficiently match a user's preferences to similar postings, giving rise to increased transactions and sales turnover.

The detailed solution to this would be to make use of unsupervised machine learning via neural networks in order to perform multi-classification of product categories. Dimensional clustering can also be used in order to match similar looking product images for the recommender system.

Metrics used to measure the performance would be accuracy using majority class and also compared to Imagenet pre-trained model performance, to be within 5 percentage points. Challenges foreseen would be lack of sufficient data, complex background noise or poor resolution images.

## Conclusion & Recommendations

Using a neural network of just 2 convolutional blocks and a dense node, the custom model was able to execute with extremely good accuracy of approximately 96% for product classification after using 50 epochs of gradual training. The high accuracy could also have been partially boosted using the the augmentation generator and regularization techniques. In comparison, the performance of the custom model came within 3 percentage points of the reknown VGG-16 pre-trained model which had a 98% accuracy.

It can be presumed that the high accuracy was partially due also to the metrics set out which was not overly abstract and also due to the consistency and purity of the base data images, which were mostly standard in each class without much variation or mislabelling. If the problem statement requirements were not demanding, a densely layered network was not necessary and a more lean model would meet the requirements quite sufficiently.

However, in detecting more granular details, a qualitiative assessment shows that the network was still not very good in distinguishing more detailed patterns. This could also be due to the fact that product surface designs are mostly unique such as shirt and bag imprints. The model limited by the richness of the underlying data would be unable to make this distinction very well due to the lack of enough image volume to combat this unique feature inbalance.

The KNN based recommendation model manages to get the product type classification correctly. It also performs relatively well with respect to dimensional,color and design features, being able to capture these elements with a good degree of likeness. For more intricate patterns and instances with the presence of a variety of different colors, the model performs not so well but it can be also due to the inherent fact that the dataset itself contains limited alternative matches and that granular patterns are supposed to be unique (such as shirt and bag motifs). Also, most products listed are expected to be unique and repeat listings are often not frequent. With just 5,000 images per category, the performance of these predictions are bound by the volume and richness of the source data.

Future recommendations to help improve on this scope would be to add more complicated product images into the mix, such as toys or jewelry, which typically do not have homogenous features and can be quite challenging to perform feature recognition on. Also, segregation within classes itself, like splitting gender related items can also be another option to explore.

Another step that could be done at the same time would be to experiment with slicing and modifying the VGG-16 model to create a hybrid structure capable of more powerful visual analysis.

## References

https://arxiv.org/pdf/1409.1556.pdf

http://cs231n.stanford.edu/reports/2017/pdfs/105.pdf
