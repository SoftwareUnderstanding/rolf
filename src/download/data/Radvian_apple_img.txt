# Welcome to my Apple Image Classification Project

In this project, I explore how we can use CNN and transfer learning to build an image classifier. The dataset consists of 1500 images scraped from Google Image's top results for 'iPhone', 'iPad', and 'Macbook'. The full dataset can be downloaded here: https://www.kaggle.com/radvian/apple-products-image-dataset.

The aim is try to create an image classifier that can differentiate 3 of the most mainstream Apple products - the iPhone, iPad, and Macbook. On the surface, they might be mistaken for each other (older iPhone design without notches can be mistaken for iPad, while newer iPad with keyboards can be mistaken for Macbook). While we can easily differentiate them, can we teach a deep learning model to do the same? And what method gives us the best accuracy? Those are the questions that will be answered in this project.

## File Descriptions

There are only 4 files in this repository (except the README and requirements file).
- The notebook is a jupyter notebook which can be run on Google Colab (with GPU for faster training). It contains step-by-step on how to create the image classifier. 
- The apple.jpg is just an image for our home page
- 'model_inception_weights.h5' is the trained weights of our deep learning model's layers. This is used to load the model in our web app.
- 'streamlit-apple.py' is the python file to deploy our web app in Streamlit.

To run the project locally, you can download this repo and type ```streamlit run streamlit-apple.py``` inside this folder's directory. 
To view the project as a deployed online web app hosted by Streamlit, we can check out this link: https://share.streamlit.io/radvian/apple-img/main/streamlit-apple.py

## Model Description

The foundational model that we use is Inception-V3 from Keras' pretrained models. However, we cut it off until 'mixed7' layer, and then add our own layers. The upper layers are used to process the image files through multiple convolutions by using pretrained weights of the model. Here are the links to learn more about the pretrained model:
- https://keras.io/api/applications/inceptionv3/
- https://arxiv.org/abs/1512.00567

We achieved 88% accuracy on validation set, and 92% accuracy on training set.
