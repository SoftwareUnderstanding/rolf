# To run, first clone the cnn directory

#### As of now, the number of layers are held static, as are the dimensions of parameters. CNN with Numpy.ipynb is composed of the network layout, while backward.py, forward.py, and utils.py hold functions used in the model.

#### The current setup is composed of input --> conv + relu --> pool --> conv + relu --> pool --> flatten --> fc1 -- fc2 (prediction). 

#### To run:
```python
# load the data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((1, 28, 28, 60000)) # (channels, rows, cols, imgs)
test_images = test_images.reshape((1, 28, 28, 10000)) # (channels, rows, cols, imgs)
```
```python
# normalize the image pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
```

```python
# initialize and store forward parameters, and first and second moments of gradient for optimization
"""
kn = conv. kernel
wn = fc. weight
"""


# forward parameters
k1 = np.random.randn(32, 1, 3, 3)
k2 = np.random.randn(64, 32, 3, 3)
w3 = np.random.randn(64, 1600) * 0.01
w4 = np.random.randn(10, 64) * 0.01

b1 = np.zeros((k1.shape[0],1))
b2 = np.zeros((k2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))

# optimization moments
v1 = np.zeros(k1.shape)
m1 = np.zeros(k1.shape)
bv1 = np.zeros(b1.shape)
bm1 = np.zeros(b1.shape)

...

v4 = np.zeros(w4.shape)
m4 = np.zeros(w4.shape)
bv4 = np.zeros(b4.shape)
bm4 = np.zeros(b4.shape)

params = [k1, k2, w3, w4, b1, b2, b3, b4]
moments = [v1,m1,bv1,bm1,v2,m2,bv2,bm2,v3,m3,bv3,bm3,v4,m4,bv4,bm4]
```

```python
# Specify number of epochs, and train over specified batch/batches (here I train over a single batch only)

cost = []       # cost per epoch
numEpochs = 10  
numLabels = 10   
batchSize = 10
Y = np.zeros((batchSize,numLabels,1))

# for each image in batch, one iteration = forward, backward, and optimization
_iter = 0 
for epoch in range(numEpochs):
    cost_ = 0 # average cost per iteration
    for img in range(batchSize):
        _iter += 1
        Y[img,train_labels[img]] = 1.                                         # one hot vector labels
        image, label = train_images[:,:,:,img],Y[img]
        loss, fp = myCNN().forwardPass(image, label, params)                     # this returns the loss and forward pass
        grads =  myCNN().backwardPass(params, loss, fp, image, label)            # this returns the gradiets w.r.t the loss
        cost_ += loss
        print("iteration ", _iter)
        if (img+1) % batchSize == 0:
            print("now optimizing: epoch ", epoch+1)
            params = myCNN().optimize(0.0001, 0.9, 0.999, 1E-7, moments, grads, params, _iter, batchSize)
            cost_ = cost_/batchSize 
            print("average cost: ", cost_)
            cost.append(cost_)
```

Information on the implementation of forward, backward, and optimization was obtained at: https://github.com/Alescontrela/Numpy-CNN/tree/master/CNN, https://github.com/Kulbear/deep-learning-coursera/blob/master/Convolutional%20Neural%20Networks/Convolution%20model%20-%20Step%20by%20Step%20-%20v1.ipynb, https://arxiv.org/abs/1412.6980.

The whole network was reformatted since the original was messy, and it wan't clear how to run several epochs with the model.

Currently, the NN is running, but the cost isn't decreasing rapidly enough. This could be due to several reasons that I will test.
1) Need more batches of data instead of one batch of 10. (going to run full dataset on cms)
2) Need to tune the optimization hyperparameters.
3) Optimization algorithm incorrectly updating parameters.
4) Backprop or forward prop wrong (both checked with the above links).
