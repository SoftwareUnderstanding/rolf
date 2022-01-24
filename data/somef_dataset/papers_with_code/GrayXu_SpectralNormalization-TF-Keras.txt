# SpectralNormalization-TF-Keras

Transfer ["IShengFang/SpectralNormalizationKeras"](https://github.com/IShengFang/SpectralNormalizationKeras) to tf.keras

I created this repo, because Keras and tf.keras are not compatible, and some libs of Keras are not supported by old Tensorflow versions.

Enviro: Tensorflow 1.12.0

# How to use?

1. Move SNorm_tf_keras.py to your code's dir.
2. Import layers like  
`from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D`
3. Use them like normal layers (but only in discriminators mentioned in the paper).

# Does Spectral Normalization work?

[View here](https://github.com/IShengFang/SpectralNormalizationKeras#result)

---

PS: if anyone found this error:`No module named 'tensorflow.keras.engine'` . Change [line 6](https://github.com/GrayXu/SpectralNormalization-TF-Keras/blob/master/SNorm_tf_keras.py#L6) to `from tensorflow.python.keras.layers import Layer`.

*[original paper on arxiv](https://arxiv.org/abs/1802.05957)*
