# parrot_vision_competition - DenseNet

DenseNet recreation based on Densely connected Convolutional Network(https://arxiv.org/pdf/1608.06993.pdf)
```
 DenseNet(num_class, nb_blocks = 4, nb_filters = 128, depth = 40, growth_rate = 12, compression = 1,
             input_shape = (150, 150), channel = 3, weight_decay = 1e-4,
             include_top = False)
 """
    num_class = number of classes of your label data,
    nb_blocks = num of stages(num of dense blocks),
    nb_filters = initial num of filters(compressed after)
    denpth = L, growth_rate = k,
    H(l) = composite function,
    compression = 1(default),
    input_shape = (150, 150)(default),
    channel = image's RGB channel,
    weight_decay = used in kernel_regularizer,
    include_top = based on keras DenseNet121, if it is true, it includes
    fully connected layer
    """
```
#### dataset : scene dataset(0 ~ 5, {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 })
https://www.kaggle.com/puneet6060/intel-image-classification

