# Optimization of Logit (Logistic Regression) Model with Lasso Regularization

The project contains the implementation of different methods for optimization of Logistic Regression Model with and without Lasso regularization. All optimization procedures aim to find model parameters, maximizing likelihood for a given dataset with optionally Lasso penalty. Additionaly, function generating random datasets is provided for testing purposes. The main purpose of this project is to show how different techniques, based on gradient descent could be implemented for Logistic Regression (Logit) Model. Please be aware that R environment allows us to create pretty simple codes, easy to use for educational purposes, but calculations are not very effective. Methods are based on consecutive steps implemented as *for loops*, what without doubt would be much less time-consuming in, for instance, C++ language. If you are going to reproduce provided functions in *C++* please remember about *RCPP* package for integration of R and C++.

Logistic Regression (Logit) Model is a basic binary regression model, widely used for scoring, e.g. in area of credit risk. The formula of model is as following:

$$log\Big(\frac{p}{1-p} \Big) = \beta_0 + \sum_{i=1}^n \beta_i x_i $$

The variable $p$ is in interval $[0,1]$ and is often interpreted as probability of some event, e.g. default of client. $(\beta_i)_{i=1}^n$ is vector of parameters and $(x_i)_{i=1}^n$ is vector of independent variables.

The considered algorithms try to minimize log-likelihood function instead of likelihood, what is equivalent. The gradient of this function is quite easy to calculate. Let start with formula for log-likelihood function:

$$log(L(\beta|X)) = log(\prod_i Pr(y_i|x_i,\beta)) = \sum_i log(Pr(y_i|x_i,\beta)) = $$
$$= \sum_i log \Big( \Big(\frac{1}{1+e^{-\beta` X_i}}\Big)^{Y_i}\Big(1 -\frac{1}{1+e^{-\beta` X_i}}\Big)^{1- Y_i} \Big) =$$
$$= \sum_i \Big( Y_i log\Big(\frac{1}{1+e^{-\beta` X_i}}\Big) + (1-Y_i)log\Big(1 -\frac{1}{1+e^{-\beta` X_i}}\Big) \Big)$$

So gradient of this function is equal to:

$$\nabla log(L(\beta|X)) = \sum_i  \Big( Y_i \nabla log\Big(\frac{1}{1+e^{-\beta` X_i}}\Big) + (1-Y_i)\nabla log\Big(1-\frac{1}{1+e^{-\beta` X_i}}\Big) \Big) = $$
$$= \sum_i \Big(Y_i -\frac{1}{1 + e^{-\beta`X_i}}\Big)X_i$$

Where $y_i$ denotes observations of binary depentent variable and $X$ denotes matrix of independent variables, $\beta$ is vector of model parameters.

Implemented methods:

- **Stochastic Gradient Descent** *optimizeLogitStochasticGradient*: standard method of optimization in stochastic version, where steps are made accordingly to information from a random batch of observations. This approach is particularly useful for big sets of data, where calculation of log-likelihood for all of them is highly time-consuming.

- **SAGA algorithm** *optimizeLogitSAGA*: method based on Defazio and Bach paper *SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives* from 2014. Available at https://arxiv.org/abs/1407.0202. Let me point out, that here algorithm is used for problem with differentiable log-likelihood, so proximal gradient operator is just an identity.

- **Proximal Gradient** for optimization with Lasso regularization *optimizeLogitLassoProximalGradient*: log-likelihood with penalty of Lasso type is no longer differentiable, but it is convex. In consequence, it is impossible to use standard stochastic gradient, thus proximal gradient should be used instead.

- **Nesterov acceleration** for for optimization with Lasso regularization *optimizeLogitLassoNesterov*: method similar to proximal gradient, but with faster convergence.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Please check the prerequisites below and make sure to install all packages and libraries before running the code.

- *MASS* package for *mvnorm* function, generating observations from multivariate gaussian distribution used in function *generateRandomLogitDataset*
- *dplyr* package for data wrangling and *%>%* operator,

### Input data structure

The input data structure is as following: the dataset should be a tibble object with each observation in separate row. The last column should contain binary variable $Y$, which is explained by the model. You could easily generate random dataset of this form using *generateRandomLogitDataset* function from *src* folder.


### Example of use

```{r}
library(MASS)
library(dplyr)

source("src/fun-getGradientLogLikelihood.R")
source("src/fun-generateRandomLogitDataset.R")
source("src/fun-optimizeLogitStochasticGradient.R")


dataset <- generateRandomLogitDataset(numberOfVariables = 7,
                                      numberOfObservations = 20000)

results <- optimizeLogitStochasticGradient(dataset = dataset,
                                           gradientStepSize = 0.01,
                                           batchSize = 500,
                                           numberOfSteps = 10000,
                                           historyFrequency = 10)
```



## Authors

* **Przemysław Ryś** - [see Github](https://github.com/PrzemyslawRys), [see Linkedin](https://www.linkedin.com/in/przemyslawrys/)

Codes were prepared for the Advanced Monte Carlo Methods course on the Faculty of Mathematics, Informatics and Mechanics, University of Warsaw and they are based on lectures provided by dr. hab. Błażej Miasojedow.