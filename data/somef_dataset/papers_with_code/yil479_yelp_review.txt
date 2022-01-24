# Yelp Business Recommendation Project

Authors : _Zihe Wang (zw2624)_, _Di Ye (dy2404)_, _Ziyao Zhang (zz2583)_, _Yinhe Lu (yl4372)_

Date: _Dec 20, 2019_

## Directories and files

Please install the required package using `pip install -r requirements.txt`. (Package of Factorization Machine can be installed using the link in reference below.)

Directory __Code__ contains developing codes in jupyter notebook. 

Directory __Notebooks__ contains generated PDFs of jupyter notebooks we used to produce our report.

Directory __Data__ contains the small datase,(large datsest is too big to upload).

Please see __final report.pdf__ for summarization of everything above and further evaluation.


## Our goal

Our business objective is to **predict rating of user to business and rank the predicted business and recommand them to the users, and we choose those users who have rated at least few businesses on the platform**. We want to make sure that we can accurately predict the ratings and recommend the businesses to user in the order they would prefer.

Summary of the objectives:
* How close our prediction is to the actual rating
* Whether the predicted ranking of businesses is the same as the actual ranking
* How do our models perform on different segmentation of user and business


## Data

Full dataset can be found here: Yelp dataset challenge https://www.yelp.com/dataset/challenge.

We construct a large dataset and a smaller dataset to work on.

We also segment the large test set from user and business dimension by three different popularities.

## Result

Please see __final_report.pdf__

## References

Recommender Systems: The Textbook, By Charu C. Aggarwal

Lecture Notes from IEORE 4571, by Dr. Brett Vintch, Columbia University

Neural Collaborative Filtering: https://arxiv.org/abs/1708.05031

Factorization Machines in Python: https://github.com/coreylynch/pyFM

Wide \& Deep Learning for Recommender Systems, Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen: https://arxiv.org/pdf/1606.07792.pdf
