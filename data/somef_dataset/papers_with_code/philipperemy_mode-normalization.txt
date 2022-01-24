# Mode Normalization (Keras)

```
ModeNormalization(axis=-1, k=2, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

Mode Normalization [Lucas Deecke, Iain Murray, Hakan Bilen - 2018](https://arxiv.org/pdf/1810.05466v1.pdf)

Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1 for K
    different modes.
    
## Arguments
- **axis**: Integer, the axis that should be normalized (typically the features axis). For instance, after a `Conv2D` layer with `data_format="channels_first"`, set `axis=1` in `ModeNormalization`.
- **k**: Integer, the number of modes of the normalization.
- **momentum**: Momentum for the moving mean and the moving variance.
- **epsilon**: Small float added to variance to avoid dividing by zero.
- **center**: If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
- **scale**: If True, multiply by `gamma`. If False, `gamma` is not used. When the next layer is linear (also e.g. `nn.relu`), this can be disabled since the scaling will be done by the next layer.
- **beta_initializer**: Initializer for the beta weight.
- **gamma_initializer**: Initializer for the gamma weight.
- **moving_mean_initializer**: Initializer for the moving mean.
- **moving_variance_initializer**: Initializer for the moving variance.
- **beta_regularizer**: Optional regularizer for the beta weight.
- **gamma_regularizer**: Optional regularizer for the gamma weight.
- **beta_constraint**: Optional constraint for the beta weight.
- **gamma_constraint**: Optional constraint for the gamma weight.


### Input shape
- Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

### Output shape
- Same shape as input.

### References
- [Mode Normalization](https://arxiv.org/pdf/1810.05466v1.pdf)

## SVHN dataset

- https://drive.google.com/file/d/1O4LIHC1ttSeE6HM0haymPeao4nkTFW8y/view?usp=sharing

## Run the tests

`pytest`
