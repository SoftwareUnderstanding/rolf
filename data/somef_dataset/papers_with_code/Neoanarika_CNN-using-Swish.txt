# CNN using swish 

Swish was introduced on Oct 2017 as an alternative
activation function to relu. Swish was found using a combinaton of exhaustive 
search and reinforcement learning. In the originial paper [1], swish had 
demostrated an improvement of top-1 classification by ImageNet by 0.9% by simply
replacing all relu activation functions with swish. Nonethless, swish is 
very easy to implement and just writing 1 line of code is enough to implement swish 
in tensorflow

## Example

```
x1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1
Y1 = x1*tf.nn.sigmoid(beta1*x1)# output is 28x28
```

# Results 

![alt text](https://github.com/Neoanarika/CNN-using-Swish/blob/master/media/loss_swish2.png)

During the inital phase of training the loss function remains, on average, the same this shows that swish suffers from poor intialisation during training, at least when using initally normal distributed weights with std_dev =0.1. 

![alt text](https://github.com/Neoanarika/CNN-using-Swish/blob/master/media/beta.png)

We were unable to replicate the results reported in the Swish paper, beta1 for us did not converge near 1 maybe because we didn't train our model long enough. 

![alt text](https://github.com/Neoanarika/CNN-using-Swish/blob/master/media/he.png)

It seems that He initilisation doesn't really help this problem. 

![alt text](https://github.com/Neoanarika/CNN-using-Swish/blob/master/media/loss_rmsprop.png)

After change from SGD to RMSprop we immediately get better results. 

# Reference 
1. Searching for Activation Functions https://arxiv.org/abs/1710.05941
