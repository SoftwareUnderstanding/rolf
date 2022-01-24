
# mixup

![Lifecycle
](https://img.shields.io/badge/lifecycle-experimental-orange.svg?style=flat)
![R 
%>%= 3.2.0](https://img.shields.io/badge/R->%3D3.2.0-blue.svg?style=flat)
![Dependencies
](https://img.shields.io/badge/dependencies-none-brightgreen.svg?style=flat)

mixup is an R package for data-augmentation inspired by 
[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

If you like mixup, give it a star, or fork it and contribute!


## Usage 

Create additional training data for toy dataset:
```r
library(mixup)

# Use builtin mtcars dataset with mtcars$am (automatic/manual) as binary target
data(mtcars)
str(mtcars)
summary(mtcars[, -9])
summary(mtcars$am)

# Strictly speaking this is 'input mixup' (see Details section below)
set.seed(42)
mtcars.mix <- mixup(mtcars[, -9], mtcars$am)
summary(mtcars.mix$x)
summary(mtcars.mix$y)

# Further info
?mixup
```


## Installation

Requires R version 3.2.0 and higher.

```r
install.packages('devtools') # Install devtools package if necessary
library(devtools)
devtools::install_github('makeyourownmaker/mixup')
```


## Details

The mixup function enlarges training sets using linear interpolations 
of features and associated labels as described in 
[https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412).

Virtual feature-target pairs are produced from randomly drawn 
feature-target pairs in the training data.  
The method is straight-forward and data-agnostic.  It should 
result in a reduction of generalisation error.

Mixup constructs additional training examples:

x' = λ * x_i + (1 - λ) * x_j, where x_i, x_j are raw input vectors

y' = λ * y_i + (1 - λ) * y_j, where y_i, y_j are one-hot label encodings

(x_i, y_i) and (x_j ,y_j) are two examples drawn at random from the training 
data, and λ ∈ [0, 1] with λ ∼ Beta(α, α) for α ∈ (0, ∞).
The mixup hyper-parameter α controls the strength of interpolation between 
feature-target pairs.

### mixup() parameters

| Parameter  | Description                                         | Notes                          |
|------------|-----------------------------------------------------|--------------------------------|
| x1         | Original features                                   | Required parameter             |
| y1         | Original labels                                     | Required parameter             |
| alpha      | Hyperparameter specifying strength of interpolation | Defaults to 1                  |
| concat     | Concatenate mixup data with original data           | Defaults to FALSE              |
| batch_size | How many mixup values to produce                    | Defaults to number of examples |

The x1 and y1 parameters must be numeric and must have equal 
numbers of examples.  Non-finite values are not permitted.
Factors should be one-hot encoded.

For now, only binary classification is supported.  Meaning y1 must contain 
only numeric 0 and 1 values.

Alpha values must be greater than or equal to zero.  Alpha equal to zero
specifies no interpolation.

The mixup function returns a two-element list containing interpolated x 
and y values.  Optionally, the original values can be concatenated with the
new values.

### Mixup with other learning methods

It is worthwhile distinguishing between mixup usage with
deep learning and other learning methods.  Mixup with deep learning 
can improve generalisation when a new mixed dataset is generated
every epoch or even better for every minibatch.  This level
of granularity may not be possible with other learning
methods.  For example, simple linear modeling may not 
benefit much from training on a single (potentially greatly
expanded) pre-mixed dataset.  This single pre-mixed dataset 
approach is sometimes referred to as 'input mixup'.

In certain situations, under-fitting can occur when conflicts
between synthetic labels of the mixed-up examples and
labels of the original training data are present.  Some learning
methods may be more prone to this under-fitting than others.

### Data augmentation as regularisation

Data augmentation is occasionally referred to as a regularisation 
technique.
Regularisation decreases a model's variance by adding prior knowledge 
(sometimes using shrinkage).
Increasing training data (using augmentation) also decreases a model's 
variance.
Data augmentation is also a form of adding prior knowledge to a model.

### Citing

If you use mixup in a scientific publication, then consider citing the original paper:

mixup: Beyond Empirical Risk Minimization

By Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz

[https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)

I have no affiliation with MIT, FAIR or any of the authors.


## Roadmap

 * Improve docs
   * Add more detailed examples
     * Different data types e.g. tabular, image etc
     * Different parameters
     * Different learning methods
 * Lint package with [goodpractice](https://cran.r-project.org/web/packages/goodpractice/index.html)
 * Add tests
 * Add support for one-hot encoded labels
 * Add label preserving option
 * Add support for mixing within the same class
   * Usually doesn't perform as well as mixing within all classes
   * May still have some utility e.g. unbalanced data sets
 * Generalise to regression problems


## Alternatives

Other implementations:
 * [pytorch from hongyi-zhang](https://github.com/hongyi-zhang/mixup)
 * [pytorch from facebookresearch](https://github.com/facebookresearch/mixup-cifar10)
 * [keras from yu4u](https://github.com/yu4u/mixup-generator)
 * [mxnet from unsky](https://github.com/unsky/mixup)


## See Also

Discussion:
 * [inference.vc](https://www.inference.vc/mixup-data-dependent-data-augmentation/)
 * [Openreview](https://openreview.net/forum?id=r1Ddp1-Rb)
 
Closely related research:
 * [Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/abs/1806.05236)
 * [MixUp as Locally Linear Out-Of-Manifold Regularization](https://arxiv.org/abs/1809.02499)

Loosely related research:
 * [Label smoothing](https://arxiv.org/pdf/1701.06548.pdf)
 * [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)


## Contributing

Pull requests are welcome.  For major changes, please open an issue first to discuss what you would like to change.


## License
[GPL-2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
