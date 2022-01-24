# Plastic-Recyclability
## About
I created this project individually as the final project for CS175 Projects in Artificial Intelligence. The goal of this project is to classify images of plastics into recyclable or maybe-recyclable plastics. 
## Problem Statement
Many people assume all plastics are recyclable, but this is not necessarily true. Different recycling centers in different areas only process certain types of plastics, while other plastics are directed towards landfill. Misguided waste disposal is problematic because recycling centers process waste in large batches, and if the batch has a certain percentage of non-recyclables, the entire batch is moved over to landfill. Therefore, an improperly recycled item could cause correctly recycled waste to end up in landfill.
## Recyclable vs Maybe Recyclable
I decided to use "maybe recyclable" instead of "non recyclable" because all plastics are technically recylable, but processing plants may or may not accept them. I labeled all plastics that are for the most part universally recyclable as "recyclable", and the rest as "maybe recyclable"
## Implementation
I tacked the problem of classifying plastic images with deep learning. To compare various models, I trained a plain convolutional network in `plain_train.ipynb`, and a residual network(https://arxiv.org/pdf/1512.03385.pdf) in `res_train.ipynb`. Training takes 20-30 minutes using Google colab GPUs. To understand the data used, look in the `data` subdirectory. Results are in image files in the `results` subdirectory.  
## Running the app
It is recommended to run this app on Google colab. The project was developed on colab, and dependency issues were present when running locally. This project should always run fine on colab, unless dependencies become deprecated. Training and inference will run faster by turning on colab's GPU support.

