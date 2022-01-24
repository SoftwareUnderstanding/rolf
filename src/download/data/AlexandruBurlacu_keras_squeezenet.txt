# Keras Squeezenet
Keras implementation of Squeeze Net architecture described in arXiv : 1602.07360v3 (https://arxiv.org/pdf/1602.07360.pdf)

It uses the so-called Eve optimizer (https://arxiv.org/pdf/1611.01505v2.pdf), implementation here: https://github.com/jayanthkoushik/sgd-feedback

## TODOs:
* ~~add data augmentation~~ (Done)
* ~~add bypasses~~ (Added simple bypasses)
* ~~experiment with Batch Normalization~~ (Parametrized inside SqueezeNetBuilder and FireModule classes)
* ~~try more levels of Dense layers~~ (Added the possibility to inject a small subnet in the model)
* Update Eve Optimizer implementation to Keras 2

![Model's Graph (using bypasses)](/model.png)

