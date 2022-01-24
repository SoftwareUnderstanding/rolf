# Real NVP 
PyTorch implementation of the Real NVP paper by Dinh et al. (1). This is an implementation of Real NVP for *density estimation*, rather than generative modelling. The model supports sample generation (backward pass through the flow) at the same computational cost as the one of density evaluation, but the code is not (yet) adapted for dealing with images. However, visualising the inverse and forward pass of two-dimensional densities is feasible, and I have recreated Figure 1 from (1) as a gif below:

<p align="center">
  <img src ="https://github.com/e-hulten/real-nvp-2d/blob/master/gifs/two_moons.gif">
</p>

All the interesting functionality is found in `model.py`.

Change the relevant parameters in `train.py` and run. E.g.,
```
# ------------ parameters ------------
continue_training = False  
gif = True # if you want to visualise the training as a gif (only for 2d densities)
density = "moons"  # set to true if you want to use the two moons dataset
n_c_layers = 10  # number of coupling layers
epochs = 200  # number of training epochs
batch_size = 100  # set batch size
lr = 5e-4  # set the learning rate of the adam optimiser
plot_interval = 1 # plot at the end of every epoch (for gif)
path = r"/Users/edvardhulten/real_nvp_2d/"  # change to your own path (unless your name is Edvard Hultén too)
distr_name = "two_moons"
duration = 0.1
# ------------------------------------
```
should yield good results on the two moons dataset in a very reasonable amount of time.

I have also added the batch norm layer presented in Appendix B of (2) by Papamakarios et al.


(1): https://arxiv.org/abs/1605.08803

(2): https://arxiv.org/abs/1705.07057
