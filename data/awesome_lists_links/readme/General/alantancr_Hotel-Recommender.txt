# Content Based Recommender

Text is an effective and widely existing form of opinion expression and evaluation by users, as shown by the large number of online review comments over tourism sites, hotels, and services. As a direct expression of users’ needs and emotions, text-based tourism data mining has the potential to transform the tourism industry.

Content-based filtering is a common approach in recommendation system. The features of the items previously rated by users and the best-matching ones are recommended. In our case, we will be transforming implicit information of hotel attributes as featuers for this recommendation engine.

## Problem Statement

In this project, the objective is to transform implicit information provided by users into explicit features for hotel recommendation system engine. There are two parts to this recommender engine using hotel attributes and reviews by users respectively to build two separate recommendation engine.

## Dataset

The dataset is the "515K Hotel Reviews Data in Europe" dataset on Kaggle (https:// [www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe](http://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)). The dataset is a csv file, containing most text. The positive and negative reviews are already in columns. The reviews are all in English, collected from Booking.com from 2015 to 2017.

The dataset contains 515738 reviews for 1492 luxury hotels in Europe.

Data files are not included this repository because of the large size. 

## Executive Summary

In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).

**There are two main data selection methods:**

Collaborative-filtering: In collaborative-filtering items are recommended, for example hotels, based on how similar your user profile is to other users’, finds the users that are most similar to you and then recommends items that they have shown a preference for. This method suffers from the so-called cold-start problem: If there is a new hotel, no-one else would’ve yet liked or watched it, so you’re not going to have this in your list of recommended hotels, even if you’d love it.

Content-based filtering: This method uses attributes of the content to recommend similar content. It doesn’t have a cold-start problem because it works through attributes or tags of the content, such as views, Wi-Fi or room types, so that new hotels can be recommended right away.

The point of content-based is that we have to know the content of both user and item. Usually you construct user-profile and item-profile using the content of shared attribute space. For example, for a movie, you represent it with the movie stars in it and the genres (using a binary coding for example).

**There are a number of popular encoding schemes but the main ones are:** 

- One-hot encoding
- Term frequency–inverse document frequency (TF-IDF) encoding
- Word embeddings

In this project, we will be discussing content-based filtering of recommender engine, turning implicit attributes into explicit features for hotel recommender engine.



## Content [¶](http://localhost:8888/notebooks/Documents/GA_Project/capstone/code/1_EDA.ipynb#Content)

1. EDA
2. Modelling
   - Based on reviews
   - Based on hotel tag attributes
3. Deployment
4. Conclusion and Recommendation

   

## Approach

I have reviewed the hotel attributes which were contributed by the user through their reviews. Firstly, i have done feature engineering and refactor attributes which are similar. 

<img src="image/ss1.png" style="zoom: 33%;" />

<img src="image/ss2.png" style="zoom: 33%;" />

Secondly, based on the features, I have created a sparse matrix which entails the presence of the attributes based on the hotels. 

<img src="image/ss3.png" style="zoom: 33%;" />

Lastly, I have computed a matrix based on cosine-similarity rule and ranked the hotel in terms of its similarity.

<img src="image/ss4.png" style="zoom: 33%;" />







## Deployment

As part of the project, I have performed using Flask on Heroku. I have learnt on the steps to take when performing an end-to-end project. This includes re-factoring of codes into functions and classes so as to easier compliation when compiling of the code for Flask. In order to deploy a model, you will need to understand what you want to achieve and re-look at the code on how you could recode to achieve that.

You may access the deployed site [here](https://hotelrecommender.herokuapp.com/). 



## Limitations and Recommendation

In this project, I have refactored the attributes based on my own research. It will be more effective if domain knowledge are provided, this would allow me to understand what are the key attributes that would be of importance and weightage should be given.


In view of gauging performance of the content-based recommender, we could potentially roll this model out for A/B testing to evaluate the performance of the model.
One hot encoding has the following limitations:

If there are too many parameters, then our matrix will be huge and make it impractical for calculations. Implicit relationships among categorical variables may be ignored.



This project can further study into exploration of using topic modelling or Word2Vec on text reviews. we can also look into using Universal Language Model Fine-tuning for Text Classification (ULMFiT). It utilizes neural networks and inductive transfer learning for text classification.(https://arxiv.org/abs/1801.06146).

## Conclusion

Big data analysis is changing the operating mode of the global tourism economy, providing tourism managers with deeper insights, and infiltrating into all aspects of tourist travels, while driving tourism innovation and development [1]. Tourism text big data mining techniques have made it possible to analyze the behaviors of tourists and realize real-time monitoring of the market.Both machine learning and current deep learning with high achievements have been greatly applied in NLP.

With the increasing of applications in the Internet, the source of data is getting more and more richer. Therefore, the various factors in the new data brings new challenges. It is also a chance to create novel methods to achieve better recommendation results. Social networks are still the focus of the recommendation research, integration methods and new algorithms will continue to appear in the future. The sound, location and other user preference information are received more and more attention. I believe the future of the recommender system will be a hot area of innovation and research.



## References



[1] Li, J.; Xu, L.; Tang, L.; Wang, S.; Li, L. Big data in tourism research: A literature review. Tour. Manag. 2018, 68, 301–323.

[2] Qin Li 1,2, Shaobo Li 3,4,* , Sen Zhang 1,2, Jie Hu 5 and Jianjun Hu 3,A Review of Text Corpus-Based Tourism Big Data Mining





