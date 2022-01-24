# Advanced Business Intelligence
Data Collection, Visualisation, User Profile, SVD, CTR, LSH, NN, Time Series


1. BeautifulSoup 

   Crawl data from an Auto safety accident website and then save data to a csv file.

   ![Screen Shot](https://github.com/ouyibei/Advanced_Business_Intelligence/blob/master/1/BeautifulSoup_Data_Collection/Screen%20Shot.png)

2. WordCloud - MarketBusket Data Visualisation

   Visualised Top 10 best-selling items.

   Dataset：MarketBasket
   
   Link：https://www.kaggle.com/dragonheir/basket-optimisation

   ![WordCloud](https://github.com/ouyibei/Advanced_Business_Intelligence/blob/master/2/WordCloud_MarketBasket/wordcloud.png)

3. Tpot - Titanic

   Performed data cleaning on the Titanic dataset. 
   
   Predicted passenger survival using the TPOT model.

   Dataset: Titanic
   
   Link: https://www.kaggle.com/c/titanic
   
   ![Titanic_Kaggle.png](https://github.com/ouyibei/Advanced_Business_Intelligence/blob/master/3/TPOT_Titanic/Titanic_Kaggle.png)

4. surprise SVD - MovieLens Ratings 
   
   Complement the rating matrix and then make predictions for a given user.
   
   Dataset: MovieLens Rating
   
   Link: https://www.kaggle.com/jneupane12/movielens/
   
   ![MovieLens.png](https://github.com/ouyibei/Advanced_Business_Intelligence/blob/master/4/SVD_MovieLens/MovieLens.png)
   
5. WDL - MovieLens Ratings

   Calculated RMSE.

   Article: Wide & Deep Learning for Recommender Systems，2016 https://arxiv.org/abs/1606.07792
   
   Tool: DeepCTR, https://github.com/shenweichen/DeepCTR
   
   Dataset: MovieLens Rating
   
   Link: https://www.kaggle.com/jneupane12/movielens/
   
   ![MovieLens.png](https://github.com/ouyibei/Advanced_Business_Intelligence/blob/master/4/SVD_MovieLens/MovieLens.png)
   
6. MiniHashLSHForest - weibo

   List top-3 similar centences of a certern centense.
   
   Dataset: Weibo news.
   
7. ResNet18 - CIFAR10

   Classification 10 calsses of pictures.   
   
   Deep Residual Learning for Image Recognition, Kaiming He, 2016 CVPR Best Paper, https://arxiv.org/abs/1512.03385v1

   Dataset: The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test      
   images. 
   
   ![CIFAR10.png](https://github.com/yibeiou/Advanced_Business_Intelligence/blob/master/7/ResNet18_CIFAR10/CIFAR10.png)

   The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images 
   from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than 
   another. Between them, the training batches contain exactly 5000 images from each class. 

   Ten classes: ariplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   
   Link: http://www.cs.toronto.edu/~kriz/cifar.html
   
8. Prophet - JetTrain

   Predict JetTrain customer amount.

   Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality,
   plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing 
   data and shifts in the trend, and typically handles outliers well.

   Prophet is open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.

   https://facebook.github.io/prophet/
   
   Dataset: Japan JetTrain
   
   ![JetTrain.png](https://github.com/yibeiou/Advanced_Business_Intelligence/blob/master/8/Prophet_JetTrain/JetTrain.png)
