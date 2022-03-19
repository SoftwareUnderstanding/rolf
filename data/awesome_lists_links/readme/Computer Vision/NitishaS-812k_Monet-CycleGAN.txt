# Monet-CycleGAN
Tensorflow and Keras implementation of a cycle GAN to transform ordinary photos to Monet paintings. 

The same notebook can be found <a href= 'https://www.kaggle.com/daenys2000/cyclegan'>here</a>.

Cycle GAN's are useful when there is an unavailability in paired training data. Training is done in a cycled fashion. This model uses the same resnet architecture which was used in the original paper. It was trained on TPU.

## Results
<img src='monet_result.png'></img>

## Applications
<ul>
  <li>Creating new pictures in the same style used by artisits that are no more.</li>
  <li>Transforming pictures</li>
  </ul>
  
## Improvements possible:
We can greatly improve the model by using more training data, we can experiment with different loss functions. We can use this cycleGAN with other datasets.

## References
1. https://www.kaggle.com/amyjang/monet-cyclegan-tutorial
2. https://keras.io/examples/generative/cyclegan/
3. https://arxiv.org/pdf/1703.10593.pdf
4. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
