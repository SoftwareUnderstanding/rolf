# Coloring in the Deep: Using Deep Learning for Image Colorization
Rohit Jawle (rjawle), Husam Salhab (hsalhab)
Martin Chu (mchu6) , Ian Maloney (rmalone1)
 
## Introduction
This paper’s objectives are to produce vibrant and realistic colorizations of grayscale photographs.  We chose this paper because it utilizes architecture we already know (CNNs) and the project seemed like an interesting one to everyone in the group. This is a supervised learning problem where  our inputs are the grey scale images and the labels are the corresponding original colored images. We hope to predict the colors for every pixel on the image, based on training our model to generate artificially coloured images that are similar to the real coloured images from a corresponding grayscale image.

## Related Work
This project is based on the Colorful Image Colorization paper linked below. The original implementation is in Caffe, but we will try to implement this in TensorFlow.
- Paper: https://arxiv.org/pdf/1603.08511.pdf?fbclid=IwAR1wo-0xutFu7ZurZJQwkZ4RDjxyaLaavW3A1tldAXyy8uLBprpSuvkS9Ps
- Github Repo: https://github.com/richzhang/colorization/tree/815b3f7808f8f2d9d683e9ed6c5b0a39bec232fb
 
## Data
The dataset we plan to use is called imagenet. Imagenet is a dataset organized according to the WordNet hierarchy of nouns. The dataset includes 1.3 million images which fall in a range of subsets called synonym sets. 
- https://github.com/foamliu/ImageNet-Downloader
- http://image-net.org/download-imageurls
How big is it? Will you need to do significant preprocessing?
- 1.3 million images (100 gb) 
- We will be truncating the dataset and using a subset of it for initial training and model development. Once the model is good-to-go, we will look into the possibility of training it on the entire dataset.
- Significant preprocessing may be needed if some images in the dataset are greyscale. These will be excluded. Also, we might change the resolution of the images during preprocessing to make them smaller, to allow for faster training.
 
## Methodology
![](https://i.imgur.com/KDZ5u2y.png)
- Our model follows a standard CNN architecture, with a few changes. It consists of 8 layer “blocks”, and a softmax layer at the end to get probability distributions. Each of these blocks consists of 2 to 3 repeated Convolution and ReLU activation layers, followed by a Batch Normalization layer. There are no Pooling layers used in this architecture.
- We’ll be training our model for 8000-10000 epochs on the ImageNet dataset. Basically batching and shuffling the data at each epoch, then doing a forward pass to obtain our probabilities. We will then calculate our loss and perform our backwards pass, doing gradient descent and updating the trainable parameters.
- Calculating the model’s loss is probably going to be the hardest part of implementation. Since there is no right or wrong with image colorization, it would be hard to tell our network how well it’s performing during training. We could use a euclidean loss to find how “different” the colored image is from the original one, but this is not optimal since it would favor “grayer” images. Instead, we will perform a multinomial classification that uses a multinomial cross-entropy loss, where we compare the probability distribution of colours for each pixel with the true colour value using KNN. 
 

 
 
## Metrics
- We plan to conduct experiments where we show colored images to participants and ask them to identify whether the image is original or was colorized using our model.
- The notion of accuracy does apply to our project. We are comparing the image that is colorized using our model to the original colored image. However, even if a colorized image does not look similar to the original image, the image can still look accurate with different colors. 
- The authors of the paper were originally hoping to see how realistic the colorized image is. They did this using a “colorization Turing Test”, where they showed the real image vs the colored image for real life participants to select which one was the fake image. 40 Participants were given 10 practice trials and 40 tests pairs.
 
## Ethics
What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
- This dataset is incredibly representative for our purposes as it contains so many different types of imagery to colorize, yet recently there has been a push to rebalance the people category in imagenet. Since this September, researchers have made a valiant effort to remedy non-imageable concepts, concept vocabulary, and the diversity of images.
- Non-imageable concepts —> These include concepts that mat not necessarily be offensive, but are not suitable for image classification. An example is race. Race can be assumed by color of the skin but that is not necessarily true. Classifying someone as one race could be offensive when they are not. This creates bias. To fix this, researchers determined the imageability of the synsets and removed any with low imageability.
- Concept Vocabulary —> this involves cleansing the wordnet dataset of offensive and derogatory terms. Researchers have manually annotated unsafe and sensitive words that could insult others
- Diversity of Images—> since images were collected from search engine queries. These have shown bias in the words that may not have a gender attached to them (such as banker). To remedy this, researchers have searched in different languages, expanded their queries, and have combined multiple search engines. The filtering of non-imageable synonym sets helps out with this as well. 
http://image-net.org/update-sep-17-2019
 
Who collected the data (and intentions?) —> was it ethically collected?
- Images were collected via querying search engines. However, even though collection methods were ethical, they have proven to show some bias towards race and gender. This was however corrected as explained above.
Ethical implications
- The model could use wrong colors, which in certain contexts would give the image a different meaning (e.g miscoloring people’s skin). We will take this into account when evaluating the success of the model.
 
## Division of labor
We have identified three key parts to the project:
- Image tools and preprocessing (e.i grayscale testing, grayscale conversion, and image resizing)
- Loss function, which is a fairly complicated task since there is no intuitive way of telling how “well” our model colors grayscale images
- Model architecture

We will start by determining how much work is involved in each of these parts and assigning them based on each person’s relevant knowledge to each part.


