# Tensorflow Pretrained Dual Path Networks (DPN)
Easy and straightforward tensorflow implementation of Dual-Path-Networks ((https://arxiv.org/abs/1707.01629) with cypw's pretrained weights, converted to tensorflow format.

# Details

* All models require 224x224 image size. 
* The input images are preprocessed by substracting the RGB mean = [ 124, 117, 104 ], and then multiplying by 0.0167.
* Axis = 3 is considered the channel axis.

# How to load weights example

```
model_type = 'dpn92'
model = dpn_model(input_shape=(224, 224, 3), model_type=model_type)
model.load_weights('{}.h5'.format(model_type))
```

# Pretrained Weights

Model    |             Weights
:--------|:-----------------------------------:
DPN-68   |https://drive.google.com/file/d/1AYeFy4JZVINqv8MOyg9CwGPT9-f3jFI3/view
DPN-92   |https://drive.google.com/file/d/1Ug_rQhxqa9x8vCoaTaLL0geTqbyHEbNk/view
DPN-98   |https://drive.google.com/file/d/1Zbrngvbqq6kvop_BCu72Y-BC0DNPtHuB/view
DPN-107  |https://drive.google.com/file/d/1_Qkpolqwg9tMqjx6yPD8-p46Hlzneh3L/view
DPN-131  |https://drive.google.com/file/d/1kxQyhsLRcIq1aeKt4vUbS1fjfqzg2bpC/view
