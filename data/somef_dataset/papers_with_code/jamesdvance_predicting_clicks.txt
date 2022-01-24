# Predicting Clicks
### A survey of different click-prediction datasets with various ml methods

Predicting clicks is hugely important to today's web. Anything personalized such as recommendations, search results or dynamic ads can be driven with a click-prediction model. Still its not a simple problem to crack. The relative quality of the model is not trivial. Over millions of impressions a company has a lot to gain from a slight improvement in their probability of click prediction. From 2016 -2021, the advances in these algorithms is substantial. This project seeks to summarize the bleeding edge of CTR prediction and how we got here.

CTR models are typically concerned with modeling complex interactions between categorical features. An online recommender system usually consists of ranking products to recommend using customer attributes. Because of this the better a model can be at modeling the complex ways those features interact, the better its recommendations will be. An example might be an app store seeing higher lift recommending based on the interaction of timestamp and app category (e.g. food ordering apps at meal times or gaming apps after school). Modeling multiple interactions can yield even more precise results (e.g. age, timestamp, and gender when recommending gaming apps). 


## Modeling Interactions

### Product Neural Networks
*Product* here refers to the mathematical product of two vectors, not the 'products' whose clicks are being predicted!
![Product Based Neural Network](images/product_neural_net.png)

Product-based neural networks (PNNs) are defined as having a layer immediately after an embedding layer which uses pairwise products between embedded features. The pairwise products can either be inner products resulting in a scaler or outer products resulting in a matrix. In addition to pairwise products creating a quadratic signal, a unit of '1' is appended to the embedding layer, meaning for each pairwise node of two embeddings, there also exists a node of a single embedding and 1, which results in a linear signal. This way the layer gets the benefit for a quadratic signal, while not ignoring a linear signal from the embeddings.  

The logic behind using pairwise products to better learn interactions is to learn weights and biases that multiplication works as a sort of AND operator between embedding vectors while addition is like an OR operator. 

The embedding layer does not use a pre-trained Factorization Machine to seed the feature vector. Rather the model learns the embeddings from scratch. Additional, several fully-connected layers are added to the end of the network to complete its learning. The paper found 3 layers, including the final activation layer, were optimal. 


#### Papers
1. [Product-Based Neural Networks For User Response Prediction](https://arxiv.org/pdf/1611.00144v1.pdf)
2. [Factorization Machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)
3. [Matrix Factorization Graphic](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
4. [Neural Factorization Machines](https://arxiv.org/pdf/1708.05027.pdf)

### Wide and Deep

### Deep Factorization Machines

Deep Factorization machines are neither factorization machines nor Neural Factorization Machines. 

Factorization machines are a heavy influence of Product Neural Networks (above) which were able to capture pairwise interactions between features. Deep Factorization Machines (DeepFM) work to improve on PNNs and FMs by capturing both low-order and high-order interactions of multiple variables. It also improves on Wide and Deep by not requiring any feature engineering. 

A little background on Factorization Machines [2]. Factorization Machines are a class of non-DL machine learning algorithm that are suited for cases of high sparsity such as purchases in recommender systems or any problem dealing with large categorical variable domains. FMs can be used to predict probability to click, a user's rating for a product, and an item's rank. They are also capable of 


#### Papers
1. [DeepFM](https://arxiv.org/pdf/1703.04247v1.pdf)
2. [Factorization Machines Article](https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db)
3. [Factorization Machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)

### Self Attention

#### Papers
1. [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

### Instance Masks

### Graph Neural Networks

### Network Normalization

### Interactions
Capturing Higher-order interaction effects are crucial to predicting CTR or click-through rate. 

#### Papers
1. [Detecting Statistical Interactions From Neural Network Weights](https://openreview.net/pdf?id=ByOfBggRZ)



### Resources

1. https://paperswithcode.com/task/click-through-rate-prediction
2. https://christophm.github.io/interpretable-ml-book/interaction.html
3. https://www.kaggle.com/hughhuyton/criteo-uplift-modelling#Uplift-Modelling


### Data Sources

1. Expedia Hotels https://www.kaggle.com/c/expedia-hotel-recommendations
2. Avito Context https://www.kaggle.com/c/avito-context-ad-clicks/data
3. Criteo https://www.kaggle.com/c/criteo-display-ad-challenge/data
4. Avazu https://www.kaggle.com/c/avazu-ctr-prediction/data
4. iPinYou https://github.com/Atomu2014/make-ipinyou-data

