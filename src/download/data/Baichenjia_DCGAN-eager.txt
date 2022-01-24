# DCGAN
DCGAN with mnist dataset.
## Reference
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- https://arxiv.org/pdf/1511.06434.pdf
- https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb

## RUN
- `python train.py` and train DCGAN for 50 epochs.
- After each epochs, we test 16 seeds to generated image and save in `generated_mnist`.
- The weights will saved in training_checkpoint.
- run 'create_gif' after training will create GIF.

## Result
![avatar](gif/dcgan.gif)






