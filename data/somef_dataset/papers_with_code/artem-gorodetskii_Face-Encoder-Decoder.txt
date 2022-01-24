# Face-Encoder-Decoder
Jupyter notebook contains tensorflow implementation of Deep Convolutional Autoencoder for face encoding and decoding. 
It was trained on a Celebrities dataset (https://www.kaggle.com/greg115/celebrities-100k) during 40 epochs on Tesla K80 GPU (Kaggle), 
training process  took about 4 hours.
# Autoencoder
The network architecture represents modified and adapted for encoding and decoding purposes architecture of the
Generative Adversarial Network proposed by A. Radford et all. (https://arxiv.org/abs/1511.06434).
# Results
The first row shows original images directly from the dataset and the second row shows images that have been passed through the autoencoder.
![GitHub Logo](/images/results.png)
