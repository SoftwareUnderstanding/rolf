# lr_range_test


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
`Tested with Python >= 3.6.8`

`lr_range_test` is a python module inspired by the Learning Rate Range Test a la Leslie N. Smith: arXiv:1803.09820v2 (https://arxiv.org/pdf/1803.09820.pdf). Provides utility functions to perform initial learn rate range testing with a given `tf.keras.Model` object and a `tf.data.Dataset` object.  This greatly reduces time in finding effective hyperparameters, as learning rate is the most influential, behind batch_size and model complexity (see paper for details).

![alt text](https://github.com/ifrit98/lr_range_test/raw/master/assets/lr_range.png "Learning Rate Range Test Results")


## Installation
```{bash}
git clone https://github.com/ifrit98/lr_range_test.git
cd lr_range_test && pip install .
```

## Demo
```{python}
import lr_range_test as lrt
lrt.demo()
```

## Usage
```{python}
import lr_range_test as lrt

ds = my_custom_dataset() # a tf.data.Dataset object
val = my_custom_val_dataset() # a tf.data.Dataset object

model = my_keras_model(lr) # custom keras model via tf.keras.Model()

# Initial (min) Learning Rate 
init_lr = 0.001
# Max learning rate to use in range test
max_lr = 2

# Perform the range test
(new_min_lr, new_max_lr) = lrt.learn_rate_range_test(
    model, ds, init_lr=init_lr, max_lr=max_lr)

# Recompile model, start with new max_lr and schedule decrease to min_lr
model = my_keras_model(lr=new_max_lr)
h = model.fit(ds, validation_data=val_ds)

# View metrics from history object from run with new lr params
lrt.plot_metrics(h)

```

