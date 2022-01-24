# Categorizing News Articles using ULMFit
I am using the AG News corpus for training a ULMFit (https://arxiv.org/abs/1801.06146) model to classify the news articles into
1. World - Category 1
2. Sports - Category 2
3. Buisness - Category 3
4. Sci/Tech - Category 4

You can read about the AG News Corpus here http://xzh.me/docs/charconvnet.pdf

I trained models with three below variants of data fields
1. Title and Description
2. Only Description
3. Only Title



**Language Model Accuracy:**

The language model's accuracy for the three variants are as follows

1. Title and Description - 43.4%
2. Only Description - 46.2%
3. Only Title - 45.5%



**Classifier Accuracy:**

The classifier's accuracy for the three variants are as follows

1. Title and Description - 92%
2. Only Description - 93%
3. Only Title - 85%

The accuracy for point 2 is slightly better than of point 1.

The accuracy for point 3 is the lowest as the title of news are mostly cryptic in nature and determining the category from the same is difficult.



**Confusion Matrix:**

Confusion matrix of the three variants are below:

1. Title and Description

![Title and description](https://github.com/anubhavmaity/Ag-News-Category-Classifier/blob/master/confusion_matrix/cm_1.png)

2. Only Description

![Only Description](https://github.com/anubhavmaity/Ag-News-Category-Classifier/blob/master/confusion_matrix/cm_2.png)

3. Only Title
 
![Only Title](https://github.com/anubhavmaity/Ag-News-Category-Classifier/blob/master/confusion_matrix/cm_3.png)



Due to space constraints, I have not pushed the models in the repository.
