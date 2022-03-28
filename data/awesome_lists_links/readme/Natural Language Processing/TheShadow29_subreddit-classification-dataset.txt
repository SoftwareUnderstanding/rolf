# subreddit-classification-dataset
Create a subreddit classifier and have extensive study

### Description

This repository contains titles scraped from different subreddits and the task is to identify which subreddit it came from. It is organized in .csv files. All data can be found in the zip file here: https://drive.google.com/file/d/1bpfz10fbHWR56W__Fcu-hcl7WqdAkVvk/view?usp=sharing

I have also segregated them into two lists: coarse grained list (17 subreddits), fine-grained list (1416 subreddits). These lists are obtained via thresholding on the number of active users, and choosing only those which support text data in the subreddit. More details can be found in the report. The train, validation, test set for the coarse and fine grained can be found here: https://drive.google.com/open?id=16WTab0JQGfPzs3MSOKJjHksz_rbLAtNX


Lastly, we also provide code using TF-IDF and ULMFiT (https://arxiv.org/abs/1801.06146). The latter is provided in the form of Jupyter notebook for easy replication. 