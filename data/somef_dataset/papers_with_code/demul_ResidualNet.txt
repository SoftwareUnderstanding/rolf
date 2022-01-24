# ResidualNet
Implementation of ResidualNet on CIFAR-10 dataset. test model with various option(bottleneck block, projection shortcut)

## 여러개의 모델을 한 스크립트 안에서 돌릴려면,

### 그래프를 따로 만들어줘야 한다.
```c
model=ResNet.ResNet(batch_size, learning_rate)
model.run(max_epoch, model_kind=1)

model2=ResNet.ResNet(batch_size, learning_rate)
model2.run(max_epoch, model_kind=2)
```

```c
class ResNet:
    def __init__(self, input_size, lr):
        self.lr = lr
        self.input_size = input_size

        self.graph = tf.Graph()
        
   def run(self, max_iter, model_kind):
        with self.graph.as_default() :
          sess = tf.Session()
```

### 혹은 변수들의 scope를 다르게 설정해준다.
```c
class ResNet:
     def build1(self, input, label, is_training=False):
         with tf.variable_scope('model1'):
                ...
                
     def run(self, max_iter, model_kind):
          sess = tf.Session()
          saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), scope='model1')
```

## 학습이 정체될 땐, Saturated Activation Function의 경우 Xavier Initialization을, ReLu의 경우 He Initialization을 해준다.
```c
Xavier = tf.Variable(tf.truncated_normal([h, w, "Channel of Input", "Channel of Output"], stddev=tf.sqrt(1/"Channel of Input")))
He = tf.Variable(tf.truncated_normal([h, w, "Channel of Input", "Channel of Output"], stddev=tf.sqrt(2/"Channel of Input")))
```

## Overfitting문제가 심각했는데, Overfitting을 방지하려면,


### 배치 정규화를 해준다. Operation를 만들때 종속변수 관리에 주의한다.
```c
BatchNorm = tf.layers.batch_normalization("Input", training="is training?")

### Operation을 만들때, 다음 명령어로 종속변수를 관리해서
### tf.GraphKeys.UPDATE_OPS라는 콜렉션에 있는 Moving Average가 업데이트, 사용되도록 해야 한다.

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
```

### Dropout을 사용한다.
```c
Dropout = tf.nn.dropout("Input", "Keep Probability")
```

### Weight Decay를 사용한다.
```c
weight = tf.Variable(tf.truncated_normal([3, 3, ch, ch]))
logit = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')

#Cross Entropy Error
loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label)))
#L2 Loss of Weight
L2_loss_of_weight = tf.nn.l2_loss(weight)
#Add
loss = tf.reduce_mean(loss + L2_loss_of_weight * "Decaying Factor")
```

```c
###Collection API를 사용해서 구현해보자###
weight1 = tf.Variable(tf.truncated_normal([3, 3, ch, ch]))
layer1 = tf.nn.conv2d(input, weight1, strides=[1, 1, 1, 1], padding='SAME')
tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(weight1), "Decaying Factor"))

weight2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch]))
layer2 = tf.nn.conv2d(layer1, weight2, strides=[1, 1, 1, 1], padding='SAME')
tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(weight2), "Decaying Factor"))

weight3 = tf.Variable(tf.truncated_normal([3, 3, ch, ch]))
logit = tf.nn.conv2d(layer2, weight3, strides=[1, 1, 1, 1], padding='SAME')
tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(weight3), "Decaying Factor"))

#Cross Entropy Error
CEE = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label)))
tf.add_to_collection('losses', CEE)

#Add ALL
loss = tf.add_n(tf.get_collection('losses'))
```

          
Reference : 




[https://stackoverflow.com/questions/41990014/load-multiple-models-in-tensorflow/41991989]


[https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/]


[https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py]

