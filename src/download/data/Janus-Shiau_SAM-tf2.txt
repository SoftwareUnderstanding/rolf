# 
# [SAM-tf2](https://github.com/Janus-Shiau/SAM-tf2)

Sharpness-Aware Minimization mechanism ("Sharpness-Aware Minimization for Eciently Improving Generalization") in Tensorflow 2.0+ style.

> Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss. This paper present empirical results that SAM improves model generalization across a variety of benchmark datasets (e.g., CIFAR-f10, 100g, ImageNet, netuning tasks) and models, yielding novel state-of-the-art performance for several. Additionally, SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specically target learning with noisy labels.

Original Paper: &nbsp; [Arxiv](https://arxiv.org/abs/2010.01412)

Offical Implementation: &nbsp; [JAX style](https://github.com/google-research/sam)

<img src="doc/algo.jpg" width="800"/>

## Usage

### Playgournd case
```bash
python lib/optimizers/sam.py
```

### SAM Wrapper
Wrap any optimizer with SAMWrapper, and use the optimize API.

```python
opt = tf.keras.optimizers.SGD(learning_rate)
opt = SAMWarpper(opt, rho=0.05)

inputs = YOUR_BATCHED_INPUTS
labels = YOUR_BATCHED_LABELS

def grad_func():
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss = loss_func(pd=pred, gt=labels)
    return pred, loss, tape

opt.optimize(grad_func, model.trainable_variables)
```
> For disable SAM, simply keep `rho=0.0` as default

Since SAM require to compute gradient twice, it's hard to make it as a real `Optimizer` class like `Lookahead` in `tensorflow_addons`.

_If anyone has good ideas to make this more simple, contributions are appreciated._

## Experiements & Benchmark

**Just providing the concept of implement this kind of mechanism in Tensorflow 2.0+.**

**I haven't conduct rigorous experiments for this implementation**

## To-do
- [ ] The testing on CIFAR-f10, Imagenet and etc.

## References

Thanks for these source codes porviding me with knowledges to complete this repository.

- google-research/sam (Official) on Github: https://github.com/google-research/sam.
