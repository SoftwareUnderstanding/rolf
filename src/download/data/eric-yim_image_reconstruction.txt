# TF 2.0 Implementation of VQVAE

See https://arxiv.org/abs/1906.00446

<img src='https://user-images.githubusercontent.com/48815706/83919525-9dbe0900-a72f-11ea-8c71-0c6ad014cdf9.png'>

<p>In the paper, the authors concatenate layers at multiple resolutions, similar to a U-net. The main difference is an explicit latent space. Here, they use the network as a VAE, by generating samples from in the latent space and upsampling.
  </p>
<p>The network itself can be repurposed for many image processing tasks: for example image restoration or image segmentation - or any task that starts with an input image and outputs some target image.</p>
<h2>Usage</h2>
<h5>File Structure</h5>
<p>Place original images in path data/originals</p>
<p>Place target images in path data/targets</p>
<p>originals and targets must have the same name for the train file to know matching pairs.</p>
<h5>Train</h5>
```
python vq_train.py
```
<h5>Viewing Results</h5>
<img src='https://user-images.githubusercontent.com/48815706/83919539-a57dad80-a72f-11ea-9ee9-c771d94362bf.png'>
```
python test_results.py --operation=view_samples --plot_type=grid
```
or
```
python test_results.py --operation=view_samples --plot_type=single
```
<h5>Creating Images from Test Set</h5>
```
python test_results.py --operation=reconstruct_test
```
<p>Notes: for image segmentation, consider changes the loss function to an IoU calculation, the network's final activation. You may also need data processing if there are multiple segmentation classes.</p>
