See issues open and closed for notes on all fastai work. Issues are used to task work and record solutions throughout the course. Below is a summary of the fastai deep learning experience from December 2018 to May 2019. 

This README.md will include all the highlights, links and techytips I have found useful throughout the course. 

# fastai: Practical Deep Learning For Coders Part 1 v3
Created/Taught by Jeremy Howard, Rachel Thomas and team @ [USFCA](https://www.usfca.edu/data-institute)

[Link to fastai course homepage](https://course.fast.ai/)

#### 0. Set-up cloud computing
I decided to use Google Cloud Platform compute with £230.58 free credit for 1 year. [This is a wonderful tutorial](https://course.fast.ai/start_gcp.html) to get started and refer back to. 
My notes are [here](https://github.com/datalass1/fastai/issues/18)

#### 1. Learn how to transfer data and notebooks to and from Google Cloud Platform instance
My notes are [here](https://github.com/datalass1/fastai/issues/20)

### Lesson 1: Image Classification

The course is taught in a top down style. Meaning that you learn how something works before you learn why it works. So let's get started. 

[Link to my notes](https://github.com/datalass1/fastai/issues/19)

[Recommended and detailed fastai notes by hiromis](https://github.com/hiromis/notes/blob/master/Lesson1.md)

Summary:
- Use Python 3 [pathlib](https://docs.python.org/3/library/pathlib.html); it is much better than use strings and used with all OS. 
- re is the module in Python that does regular expressions - things that's really useful for extracting text.
- For a computer vision task use `ImageDataBunch` fastai function.  In fastai, everything you model with is going to be a DataBunch object. Basically DataBunch object contains 2 or 3 datasets - it contains your training data, validation data, and optionally test data.
- a GPU has to apply the exact same instruction to a whole bunch of things at the same time in order to be fast. If the images are different shapes and sizes, you can't do that. So we actually have to make all of the images the same shape and size `size=224`. Models are designed so that the final layer is of size 7 by 7, so we actually want something where if you go 7 times 2 a bunch of times (224 = 7*2^5)
- **DATA AUGMENTATION**: randomises cropping, flips image, resizing, and padding.
- **NORMALIZATION**: channels of an image have a mean of zero and a standard deviation of 1. 
- Use a **Learner** to set up a model, such as `learn = create_cnn(data, models.resnet34, metrics=error_rate)`
- **ARCHITECTURE** aka the model; a mathematical framework essentially. ResNet34 and ResNet50 architectures were trained on about one and a half million pictures on ImageNet data. So we can download those pre-trained weights so we start with a model that knows something: **PRE-TRAINED MODEL and TRANSFER LEARNING**. We can also start from scratch, see lesson 7. 
- Fit a model using `learn.fit_one_cycle(4)`. This number, 4, basically decides how many times do we show the dataset to the model so that it can learn from it. 
- Results: **LOSS FUNCTION** is something that tells you how good was your prediction. Specifically that means if you predicted one class of cat with great confidence, but actually you were wrong, then that's going to have a high loss because you were very confident about the wrong answer. 
- Check out a confusion matrix, or if you have lots of classes try fastai's 'most confused': `interp.most_confused(min_val=2)`
- How do you make a model better? **Unfreeze** (i.e. train the whole thing), and use **learning rate finder**
```
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```
- Additionally a model with more layers, ResNet50, could be used to improve the model. 

### Lesson 2: Deeper Dive into Computer Vision: Data cleaning and production; SGD from scratch

[Link to my note(s](https://github.com/datalass1/fastai/issues/21)

[Recommended and detailed fastai notes by hiromis](https://github.com/hiromis/notes/blob/master/Lesson2.md)

Summary:
- "If your're stuck, keep going"
- The Google Image javaScipt to collect your ow imagery is inspired by Adrian Rosebrock on pyimagesearch. 
- [gi2ds is an amazing tool](https://github.com/toffebjorkskog/ml-tools/blob/master/gi2ds.md)
- TO BE CONTINUED...


### Recommended journal papers
- Leslie N. Smith, [A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS: PART 1 – LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY](https://arxiv.org/pdf/1803.09820.pdf)
- Leslie N. Smith, [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
- Matthew D. Zeiler et al, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)


### Resources and interative demos
- [Neural Network playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-gauss&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.44189&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=regression&initZero=false&hideText=false)
