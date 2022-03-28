[![](https://img.shields.io/badge/Made%20With-R-9cf)](https://github.com/agnesdeng/misle)
[![](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/agnesdeng/misle)
[![](https://img.shields.io/badge/Lifecycle-Experimental-ff69b4)](https://github.com/agnesdeng/misle)
# misle
Multiple imputation through statistical learning

The R package `misle` is built using TensorFlow™, which enables fast numerical computation and thus provides a solution for large-scale multiple imputation.

`misle` is still at the early stage of development so lots of work have to be done before it is officially released. 

## Current version
1. **multiple imputation by XGboost**
2. **multiple imputation by variational autoencoders**
3. **multiple imputation by denoising autoencoders(with dropout)**


## Recent updates and added features
- Mixgb imputer with GPU support.
- Tensorflow 2 compatible.
- Imputation models can be saved to impute new unseen data.
- Fixed bugs related to impute.new( ) function for mixgb imputer.
- A data cleaning function for users to roughly check their data before feeding in an imputer.
- Added several error and warning messages related to user errors for mixgb.


## Under development 
- Fixing bugs and add warning messages user errors for midae and mivae.
- Migrate TensorFlow 1 code to TensorFlow 2
- Writing up documentation and vignette
- Pretune hyperparamters for imputers
- Visual diagnostic for imputation results




## Install `misle` (It may have compatibility issues with python version/tensorflow version) 

Our package  `misle`  is built using `tensorflow`.  If P package  `tensorflow`  has been properly installed, users can directly install the newest version of  `misle`  from GitHub.

``` r
devtools::install_github("agnesdeng/misle")
library(misle)
```

If  `tensorflow` has not been installed, we recommend to use virtual environment to install it.

``` r
library(reticulate)

#By default, python package tensorflow would be installed in the virtual environment named 'r-reticulate' 
virtualenv_install(packages = c("tensorflow==1.14.0"))

#Install tensorflow R package
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(method="virtualenv",version="1.14.0",envname = "r-reticulate")

#Install misle 
devtools::install_github("agnesdeng/misle")
library(misle)

```

## Install `mixgb` (Highly recommend)
If users only want to use multiple imputation through XGBoost, please install this simplified R package `mixgb` instead.


```r
devtools::install_github("agnesdeng/mixgb")
library(mixgb)
```
## Example: multiple imputation through Denoising autoencoder
```r
#a data frame (consists of numeric/binary/multicalss variables) with NAs
withNA.df=createNA(adult,p=0.3)

#create a variational autoencoder imputer with your choice of settings or leave it as default
MIDAE=Midae$new(withNA.df,iteration=20,input_drop=0.2,hidden_drop=0.3,n_h=4L)

#training
MIDAE$train()

#impute m datasets
imputed.data=MIDAE$impute(m = 5)

#the 2nd imputed dataset
imputed.data[[2]]
```
### Impute new unseen data using Midae

```r
n=nrow(adult)
idx=sample(1:n, size = round(0.7*n), replace=FALSE)

train.df=adult[idx,]
test.df=adult[-idx,]

trainNA.df=createNA(train.df,p=0.3)
testNA.df=createNA(test.df,p=0.3)

#use training data to train the models
MIDAE<-Midae$new(trainNA.df,n_h=4L,iteration = 20,batch_size = 500)
MIDAE$train()

#obtain 5 imputed datasets for training data
imputed.data=MIDAE$impute(m = 5)

#use this imputer to impute test data 
imputed.new=MIDAE$impute.new(newdata=testNA.df,m = 5)

```


## Example: multiple imputation through Variational autoencoder
```r
#a data frame (consists of  numeric/binary/multicalss variables) with NAs
withNA.df=createNA(adult,p=0.3)

#create a variational autoencoder imputer with your choice of settings or leave it as default
MIVAE=Mivae$new(withNA.df,iteration=20,input_drop=0.2,hidden_drop=0.3,n_h=4L)

#training
MIVAE$train()

#impute m datasets
imputed.data=MIVAE$impute(m = 5)

#the 2nd imputed dataset
imputed.data[[2]]
```

### Impute new unseen data using Mivae

```r
n=nrow(adult)
idx=sample(1:n, size = round(0.7*n), replace=FALSE)

train.df=adult[idx,]
test.df=adult[-idx,]

trainNA.df=createNA(train.df,p=0.3)
testNA.df=createNA(test.df,p=0.3)

#use training data to train the models
MIVAE<-Mivae$new(trainNA.df,n_h=4L,iteration = 20,batch_size = 500)
MIVAE$train()

#obtain 5 imputed datasets for training data
imputed.data=MIVAE$impute(m = 5)

#use this imputer to impute test data 
imputed.new=MIVAE$impute.new(newdata=testNA.df,m = 5)

```

## Example: multiple imputation through XGBoost

We first load the NHANES dataset from the R package "hexbin".
``` r
library(hexbin)
data("NHANES")
```

Create 30% MCAR missing data.
``` r
#a dataframe (consists of numeric/binary/multicalss variables) with NAs
withNA.df<-createNA(NHANES,p=0.3)
```

Create an Mixgb imputer with your choice of settings or leave it as default.

Note that users do not need to convert the data frame into dgCMatrix or one-hot coding themselves. Ths imputer will convert it automatically for you. The type of variables should be one of the following: numeric, integer, or factor (binary/multiclass).
``` r
MIXGB<-Mixgb$new(withNA.df,pmm.type="auto",pmm.k = 5)
```

Use this imputer to obtain m imputed datasets.
``` r
mixgb.data<-MIXGB$impute(m=5)
``` 

Users can change the values for hyperparameters in an imputer. The default values are as follows.

``` r
MIXGB<-Mixgb$new(data=.., nrounds=50,max_depth=6,gamma=0.1,eta=0.3,nthread=4,early_stopping_rounds=10,colsample_bytree=1,min_child_weight=1,subsample=1,pmm.k=5,pmm.type="auto",pmm.link="logit",scale_pos_weight=1,initial.imp="random",tree_method="auto",gpu_id=0,predictor="auto",print_every_n = 10L,verbose=0)
```


### Data cleaning before feeding in the imputer

It is highly recommended to clean and check your data before feeding in the imputer. Here are some common issues:

- Data should be a data frame.
- ID should be removed 
- Missing values should be coded as NA not NaN
- Inf or -Inf are not allowed
- Empty cells should be coded as NA or sensible values
- Variables of "character" type should be converted to "factor" instead
- Variables of "factor" type should have at least two levels
```

cleanWithNA.df<-data_clean(rawdata=rawWithNA.df)
```


### Impute new unseen data using Mixgb
First we can split a dataset as training data and test data.
``` r
set.seed(2021)
n=nrow(iris)
idx=sample(1:n, size = round(0.7*n), replace=FALSE)

train.df=iris[idx,]
test.df=iris[-idx,]
```

Since the original data doesn't have any missing value, we create some.
``` r
trainNA.df=createNA(data=train.df,p=0.3)
testNA.df=createNA(data=test.df,p=0.3)
```

We can use the training data (with missing values) to obtain m imputed datasets. Imputed datasets, the models used in training processes and some parameters are saved in the object `mixgb.obj`.

``` r
MIXGB=Mixgb.train$new(data=trainNA.df)
mixgb.obj=MIXGB$impute(m=5)
```

By default, an ensemble of imputation models for all variables in the training dataset will be saved in the object  `mixgb.obj`. This is convenient when we do not know which variables of the future unseen data have missing values. However, this process would take longer time and space.

If users are confident that only certain variables of future data will have missing values, they can choose to specify these variables to speed up the process. Users can either use the indices or the names of the variables in the argument `save.vars`. Models for variables with missing values in the training data and those specified in `save.vars` will be saved.

``` r
MIXGB=Mixgb.train$new(data=trainNA.df)
mixgb.obj=MIXGB$impute(m=5,save.vars=c(1,3,5))

#alternatively, specify the names of variables
mixgb.obj=MIXGB$impute(m=5,save.vars=c("Sepal.Length","Petal.Length","Species"))
```

We can now use this object to impute new unseen data by using the function `impute.new( )`.  If PMM is applied, predicted values of missing entries in the new dataset are matched with training data by default. 

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df)
test.impute
```
Users can choose to match with the new dataset instead by setting `pmm.new = TRUE`.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE)
test.impute
```
Users can also set the number of donors for PMM when impute the new dataset. If  `pmm.k` is not set here, it will use the saved parameter value from the training object  `mixgb.obj`.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE, pmm.k=3)
test.impute
```

Similarly, users can set the number of imputed datasets `m`.  Note that this value has to be smaller than the one set in the training object. If it is not specified, it will use the same `m` value as the training object.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE, m=4)
test.impute
```




## Expected to be done in 2021-2022


- **Simulation studies**


   to show whether multiple imputation using statistical learning (machine learning) techniques will lead to statistical valid inference. 

- **Visual diagnostics**


   includes plotting functions for users to check whether the imputed values are sensible


## Reference
JJ Allaire and Yuan Tang (2019). tensorflow: R Interface to 'TensorFlow'. R package version 2.0.0. https://github.com/rstudio/tensorflow

Tianqi Chen, Tong He, Michael Benesty, Vadim Khotilovich, Yuan Tang, Hyunsu Cho, Kailong Chen,Rory Mitchell, Ignacio Cano, Tianyi Zhou, Mu Li,Junyuan Xie, Min Lin, Yifeng Geng and Yutian Li (2019). xgboost: Extreme Gradient Boosting. R package version 0.90.0.2. https://CRAN.R-project.org/package=xgboost

JJ Allaire and François Chollet (2019). keras: R Interface to 'Keras'. R package version 2.2.4.1.9001. https://keras.rstudio.com

Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys (1. print. ed.). New York [u.a.]: Wiley.

Vincent, P., Larochelle, H., Bengio, Y., \& Manzagol, P. (Jul 5, 2008). Extracting and composing robust features with denoising autoencoders. Paper presented at the 1096-1103. doi:10.1145/1390156.1390294 Retrieved from http://dl.acm.org/citation.cfm?id=1390294

Gal, Y., \& Ghahramani, Z. (2015). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. Retrieved from https://arxiv.org/abs/1506.02142

Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. NIPS.

Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NIPS.
 
Kingma, D. P., \& Welling, M. (2013). Auto-encoding variational bayes. Retrieved from https://arxiv.org/abs/1312.6114

Alex Stenlake & Ranjit Lall. Python Package MIDAS: Multiple Imputation with Denoising Autoencoders https://github.com/Oracen/MIDAS
