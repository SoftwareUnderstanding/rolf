# Rigid Batch Norm


### Data
1. Mnist
2. Cifar10
3. Cifar100
4. Fashion mnist


### MODEL
1. simple
2. small resnet


### Criteria
1. log loss
2. acc (= top 1 erorr)


### FIXED
1. relu
2. sgd
3. xavier


### IDEA 1
- There is a possibility that outliers exist, too big or small value.
If it is, batch mean is corrupted. So I think it is possible to
eleminate the effect of outliers when calculating batch mean by
calculating mean twice. At first time, mean and variance are calculated
to whole batch. Then, we can get normalized value and seperate data
whether its value is in bound or not. Then we can get mean and variance
derive from values with in bound. The original idea is to use these mean and variance
instead of whole batch mean and variance.
- There are derivatives backed to the above idea
1) Keep original variance
2) Clip re-normalized values to mitigate the effect of unpredictable values
3) Use exceeded values which is out of bound as a regularizing term by square

- But the idea and derivatives are not effective. I cannot get remarkable result
compared to original batch norm. I suspect this is because
1) There seldomly exist outliers. So these process has small effect and redundant.
Honestly, it reduce fast as first moving average initializer effect disappers
2) At first time of training, if we take 3) as a regularizing term, it's value
is too big to concentrate on minimize loss. So network take a time to lower
it's term.
3) It cannot permit large learning rate like batch norm if we did not handle
a regularizer as we apply a regularizer

### IDEA 2
- If we can assume the distribution of batch values are normal,
it is cheap to calcualte kullbeck leibler divergence of these values and
standard normal distribution, N(0, 1). Then, I thought it is possible to keep
original values and use kullbeck leibler divergence to induce original values
not to far from away standard normal, which can be attained if we use batch norm
with an assumption that it is normal. But this cannot output better result than batch norm. This is because
1) It cannot be assured that each layers batch values is derive from normal distribution
2) There is a limitation of a regularizer. kullbeck leibler divergence does not
dimnish below at some value.
3) If we use it, it is reasonable that the range of initializing values are
relevant to the number of node. This is because
the weighted summation of normal distributions is also normal distribtion.
4) In addition, we cannot guarantee output of activation is normal. Honestly,
absolutely not with any activation.
5) One property of batch norm is that gradients are stablized because it is
normalized and the existence of terms gamma acts as a gate in some sense.
The existence of a term gamma in the gradient of that is assured. So we suspect
gradients are stablized becuase a term gamma is affected to all channel values
when we use cnn.
