# simple-wavenet
Simple script for defining a WaveNet model using tensorflow and python - in 150 lines!

Example use:

```
# define input tensor
inputs = tf.placeholder(tf.float32, shape=(<batch_size>, <num_time_samples>, <num_input_channels>))

# define wavenet model
W = Wavenet1D(num_input_channels)
W.define_variables()

# get output tensor
outputs = W.define_graph(inputs)

# output shape = (<batch_size>, <num_time_samples>, <num_hidden_channels>)
```
## References

Oord, Aaron van den; Dieleman, Sander; Zen, Heiga; Simonyan, Karen; Vinyals, Oriol; Graves, Alex; Kalchbrenner, Nal; Senior, Andrew; Kavukcuoglu, Koray (2016-09-12). [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499). arXiv:1609.03499
