# A Tensorflow 2.0 Version of Reconstruction from Latent Space

<br>see https://arxiv.org/abs/1906.00446
<br>see https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

I use Vector Quantizer to recreate 1-minute distributions of trading data. This distribution shows the volatility and positive/negative movement within a minute. If financial data can be simulated and datasets broadened, it could help to improve accuracy on existing regression or even reinforcement learning models.

Here, the model encodes an original into a latent space. Then the model decodes the latent space back to a distribution. Per the article, to create new distributions, another model must be made to find the relationships between latent space values. Those latent space value would then feed into the decoder.



<img src='https://user-images.githubusercontent.com/48815706/76150591-d50a1480-6071-11ea-9186-eee138d59bdb.png'>
