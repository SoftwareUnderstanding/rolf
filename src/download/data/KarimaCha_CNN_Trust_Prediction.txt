
---
<h2>Project Description</h2>

<h4>Who Do We Trust?  Predicting Social Perceptions from Facial Images</h4>

Human judgments of trustworthiness from facial appearance strongly influence social decision-making in various fields, ranging from consumer behavior over employee & partner selection to political elections and jury decisions. 

This project aims to predict trustworthiness and other social perceptions from human faces using different machine learning models, including Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs). The models are trained on the *10k US Adult Face Dataset* (http://www.wilmabainbridge.com/facememorability2.html), which contains 2222 human face images along with human ratings of trustworthiness and other social attributes. 

Best preditive performance is achieved by a CNN based on the Resnet50 architecture (https://arxiv.org/abs/1512.03385) and pre-trained on facial images (https://github.com/rcmalli/keras-vggface). This model yields a correlation of 80% between true and predicted ratings of trustworthiness, and 66-88% correlation for other social attributes like friendliness, sociability, attractiveness, emotionality and intelligence. Furthermore, a deconvolutional approach is taken to visualize facial features contributing to the perception of these social attributes.

Technical Highlights: ``Python`` / ``Pandas`` / ``NumPy`` / ``Sklearn`` / ``Convolutional Neural Networks`` ``(Deep Learning)`` / ``Support Vector Machines`` / ``Exploratory Data Analysis`` / ``Unsupervised Learning`` / ``TensorFlow`` / ``Keras`` / ``Matplotlib`` / ``Seaborn`` / ``Google Cloud Services``

<h4>Application Example:</h4>

![Application Example](https://github.com/KarimaCha/CNN_Trust_Prediction/blob/master/Application_Example_Image.png)

---
<h2>Content of Repository:</h2>

The project is divided into several jupyter notebooks for easier workflow:

- <b>1_Explore_Social_Ratings.ipynb</b>: includes data cleaning, data exploration and principal compontent analysis (PCA) of the social ratings associated with the face images. 

- <b>2_Explore_Facial_Images.ipynb</b>: includes data cleaning, data exploration and preprocessing (resizing, padding, conversion to grayscale) of the face images.

- <b>3_Preprocess_images.ipynb</b>: includes further preprocessing (face extraction, PCA) of the face images for predictive modeling.

- <b>4_Support_Vector_Machine.ipynb</b>: includes predictive modeling of social ratings using Support Vector Machines (SVMs) and grid search cross validation.

- <b>5_CNN_Train.ipynb</b>: includes predictive modeling of social ratings using Convolutional Neural Networks (CNNs) and transfer learning.

- <b>6_CNN_Visualize.ipynb</b>: visualizes the most important facial features for predicting trustworthiness using methods like GradCAM and Guided Backpropagation. 

- <b>7_Summarize_Results.ipynb</b>: summarizes predictive modeling results for the Support Vector Machines (SVMs) and Convolutional Neural Networks (CNNs).

- <b>8_Application.ipynb</b>: includes an application which predicts six social ratings (trustworthy, friendly, sociable, attractive, emotional, intelligent) from human face images.
