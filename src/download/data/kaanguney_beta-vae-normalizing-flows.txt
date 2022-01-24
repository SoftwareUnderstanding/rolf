# Beta-VAE-Normalizing-Flows

### Overview
* Beta-VAE-Normalizing-Flows
  * Beta VAE connected to a normalizing flow of selection

* Noisy moons
  * [Initial test data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

* Thoracic surgery 
  * [Current test data](https://www.kaggle.com/sid321axn/thoraric-surgery)

* Beta-VAE 
  * Variational autoencoder algorithm extended with a beta parameter to put implicit pressure on the learnt posterior
  * [Find out more](https://paperswithcode.com/method/beta-vae)

### Updates 
* [Preprocessing](https://github.com/kaanguney/normalizing_flows/tree/main/scripts/preprocessing) currently supports a dataset called `prostate.xls`. Now supports `ThoracicSurgery.csv` as well.
* Refer to [beta-vae-normalizing-flows](https://github.com/kaanguney/normalizing_flows/tree/main/beta-vae-normalizing-flows) for latest results as of date of this commit.
  
### Performance Evaluation 
  * KL Divergence
  * MAE
  * Cross Entropy
  * [2-dimensional Kolmogorov-Smirnov Test](https://github.com/syrte/ndtest/blob/master/ndtest.py)

### References
* Rezende, D. J., & Mohamed, S. (2015). [Variational Inference with Normalizing Flows.](https://arxiv.org/abs/1505.05770v6)
* Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2019). [Normalizing Flows: An Introduction and Review of Current Methods.](https://arxiv.org/abs/1908.09257v4)
* [Probabilistic Deep Learning with TensorFlow 2 by Imperial College London](https://www.coursera.org/learn/probabilistic-deep-learning-with-tensorflow2)
* Blog posts
  * [Eric Jang](https://github.com/ericjang/normalizing-flows-tutorial)
  * [Lilian Weng](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
