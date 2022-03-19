
# DCGAN IMPLEMENTION
This is an implementation of the DCGAN from scratch on the MNIST dataset.
To read more on the DCGAN, refer to the original paper - https://arxiv.org/pdf/1511.06434

  - You can implement script on Jyupter Notebook for faster and efficient results [or]
  - Implement it on your personal computer with high end GPU services. The code for the same is not included in the script as it is CPU working edition.

## Getting Started

### Packages to install
* Tensorflow - version 2.0+.
* Keras - with tensorflow in the backend & importing layers and model.
* NumPy - Mathematical computations.
* PIL - Image manipulation.
* Matplotlib - For the plotting of images.
* tfutils - Utility for Keras and Tensorflow.


```sh
#Installation - tfutils
$ pip3 install git+https://github.com/am1tyadav/tfutils.git
```

##  Breaking down the code
> line 46 in the `'Generator'` block
```  
Dense(256, activation='relu', input_shape=(1,)), Reshape((1,1,256)),
```
The original paper of DCGAN has an input dimention vector of 128.  But we are using ```input_shape=(1,))``` for simplification.
> line 51 in the `'Generator'` block
```
Conv2DTranspose(1, 4, activation='sigmoid')
```
1 - The channel information
> line 66
```
plt.imshow(np.reshape(generated_image,(28,28)), cmap='binary')
```
The tensor is reshaped into ```generated_image,(28,28)``` removing the channel as it is a black-&-white image using pyplot.
> line 90
```
true_example = x[int(batch_size/2)*step:int(batch_size/2)*(step+1)]
```
The batch size is divided into half as one half of the examples are true and the other half are going to be generated.
> line 109 and 110 respectively
```
 loss, _ = gan.train_on_batch(noise, np.ones(int(batch_size/2)))
_, acc = discriminator.evaluate(xbatch, ybatch, verbose=False)
```
The Generator is trained to get an output = 1 for all the images.
To find the accuracy of the batch. As and when the Generator does well the accuracy of the Discriminator reduces.

## Built with
* Python
## Author
* **Nikita Kodkany** - *Student*

## Reference
* UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS - https://arxiv.org/pdf/1511.06434
