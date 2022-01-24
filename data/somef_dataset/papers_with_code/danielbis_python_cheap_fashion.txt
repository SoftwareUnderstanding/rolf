Overview:

We used Google’s Inception v3 model (https://arxiv.org/pdf/1409.4842.pdf) to process the images and extract their features. Inception is designed to label images among 1000 categories. We used it, but instead of passing our input through the final layer, we extract the features from the last pooling layer (pool3:0). It gives us a 2048 dimensional vector of most signifiant features. After that we cluster the images into multidimensional space using K-Means Clustering and find the nearest neighbors, which are essentially our most similar pictures.  

After that we populated SQL with the data about the matches: prices, url, titles and paths to the jpeg files. We created a simple FLASK based web application that serves as a demo for our project. 

One of the challenges that we encountered was gathering the data. Furthermore, data obtained varied in size and needed processing. Big variance in backgrounds and models wearing t-shirts affected our results, but more on that in the 'Room for improvement’ section. 

Websites Scraped for matches:
*  Zappos
*  Zummies
*  Asos
*  Shop.css
*  Zalando


Room for improvement:

We are generally getting a low percentage of similarity between the original and the match results. This is primarily due to the fact that we did not train the machine learning program for t-shirts, but used pre-trained Inception v3 Architecture. Inception v3 was made by Google Research Team. It is designed to label pictures into one of thousand categories. Not specifically for comparing t-shirts. We could retrain the model, but that required much larger dataset that we did not have. 

Another way to improve the accuracy would be to use different, preferably smaller architecture, preferably using auto encoders, as latest research proves that these work very well in similar problems. 
We were unable to scrap enough images or find a useable data set. Additionally, the program is doing k-means clustering for the entire image, not just specifically the t-shirts in the photo. This is a result of the variation in background and context of the shirt from photos from various sources we scraped. 




List of libraries used:

```
Scrapy
Annoy
nltk
Keras
Flask
Instagram_Scraper
Psutil
Numpy
Urllib
Tensorflow
AnnoyIndex
Spatial
```
Sources:

http://douglasduhaime.com/posts/identifying-similar-images-with-tensorflow.html
https://www.tensorflow.org/tutorials/image_recognition
https://stackoverflow.com/questions/34809795/tensorflow-return-similar-images
Rethinking the Inception Architecture for Computer Vision, (Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna), https://arxiv.org/abs/1512.00567 

Authors: Daniel Bis, Steven Centeno
