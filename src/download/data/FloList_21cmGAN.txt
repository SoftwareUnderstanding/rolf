# 21cmGAN
This is the Tensorflow implementation of the code 21cmGAN, which generates 2D tomographic samples of the 21cm brightness temperature of HI gas at a resolution of (up to) 32 x 256 pixels during the Epoch of Reionisation between redshifts  <img src="https://render.githubusercontent.com/render/math?math=z = 6 - 15">. The paper can be found here: [https://doi.org/10.1093/mnras/staa523](https://doi.org/10.1093/mnras/staa523) ([arXiv:2002.07940](https://arxiv.org/pdf/2002.07940.pdf)). The neural network is a progressively growing generative adversarial network (PGGAN, T. Karras, T. Aila, S. Laine, J. Lehtinen, *Progressive Growing of GANs for Improved Quality, Stability, and Variation*, [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)).

![PGGAN](https://github.com/FloList/21cmGAN/blob/master/pngs/PGGAN_sketch.png)

The 21cmGAN code makes use of code snippets from the following sources:

 - https://github.com/tkarras/progressive_growing_of_gans
 - https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow
 - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_cnn.py
 - https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/master/ops.py

*Author*: Florian List (Sydney Institute for Astronomy, School of Physics, A28, The University of Sydney, NSW 2006, Australia).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
   
For any queries, please contact me at flis0155 at uni dot sydney dot edu dot au.

The training data is taken from the 21SSD catalogue (B. Semelin, E. Eames, F. Bolgar, M. Caillat, *21SSD: a public data base of simulated 21-cm signals from the epoch of reionization*, 2017, [MNRAS 472, 4508](http://academic.oup.com/mnras/article/472/4/4508/4104651/21SSD-a-public-data-base-of-simulated-21cm-signals)).

# Requirements
21cmGAN has been tested with `Tensorflow 1.13.1`, `Numpy 1.15.4`, `Scipy 1.1.0`, `Matplotlib 3.1.0`. Moreover, we use `Scikit-image 0.14.0` for resizing images, `Pynverse 0.1.4.4` for numerically inverting the scaling of the brightness temperature (this is not needed if you scale the data with a function that has an analytic inverse function), and `h5py 2.8.0` for loading HDF5 files.

# Generating samples using the trained PGGAN 

Using the trained PGGAN for the generation of tomographic samples is easy. A tutorial that shows how to do it is provided as a Jupyter notebook, named `21cmGAN_tutorial.ipynb`. The three parameters that you can adjust are:

- X-ray emissivity <img src="https://render.githubusercontent.com/render/math?math=f_X = (0.1 - 10.0)">
- Fraction of hard X-rays <img src="https://render.githubusercontent.com/render/math?math=r_{h/s} = (0.0 - 1.0)">
- Lyman-band emissivity <img src="https://render.githubusercontent.com/render/math?math=f_\alpha = (0.5 - 2.0)">

The PGGAN was trained on parameters within the ranges given in brackets.

Since the trained neural network is too large to host it on Github as a whole, the file is split up into several parts. Run

		cat trained.ckpt-440000.tar.xz.part* > trained.ckpt-440000.tar.xz
Then, extract the archive trained.ckpt-440000.tar.xz in the same folder where `checkpoint`, `trained.ckpt-440000.index`, and `trained.ckpt-440000.meta` are saved.
		

# Retraining 21cmGAN

## Get the data
### 21SSD data
If you want to take the 21SSD data that we used for training 21cmGAN, you can download the raw data from here https://21ssd.obspm.fr/browse/ (after registering here: https://21ssd.obspm.fr).
For convenience, we provide a shell script (`download_data.sh`) and a python script (`process_high_res_data.py`) that can be used for downloading the lightcone files and creating slices. For the results reported in the 21cmGAN paper, we used one HDF5 file for each progression stage of the PGGAN, generated with the script `create_downscaled_version.py`. If you prefer using TFRecord files, have a look at the script `save_as_TFRecord.py`.


### Data from another data base
If you want to train 21cmGAN on data from another data base, note the following: 

- if you use one Numpy file containing all the data, the code expects the parameters and images to be fields of a dictionary, i.e.

		data = np.load("....")  
		X = data[()]["params"]  
		Y = data[()]["data"]
where X has shape n_slices x n_params, and Y has shape n_slices x H x W.

- if you use one HDF5 file for each stage, the code expects the following format:
		
		with h5.File("...", 'r') as hf:  
			X = np.asarray(hf["params"])
			Y = np.asarray(hf["data"])  
			
- if you use one TFRecord file for each stage, the code expects two variable length features as follows:

		features = tf.parse_single_example(serialised,  
			       features={  
				'params_raw': tf.VarLenFeature(tf.float32),  
                      'image_raw': tf.VarLenFeature(tf.float32)})


If your images have a different aspect ratio from 1 : 8, you will need to manually go through the code and change this wherever it occurs (one of the reasons for this inflexible behaviour is that the two final convolutional layers with kernel sizes 1x4 and 1x5 with valid padding at the end of the discriminator are chosen such that a single number (1 x 1) results.
Moreover, check the functions "parse_function" and "parse_function_tf" in `ops.py` that might be adjusted, in particular "scale_pars" and "scale_pars_tf", as well as the plotting functions in `utils.py` some of which are tailored for the 21SSD catalogue (legends, etc.).

## Train 21cmGAN
The script for training 21cmGAN is `main.py`.  First, adjust the hyperparameters (fields of the dictionary "par"), then run the script.

## Evaluate 21cmGAN
A workflow for producing the plots in the 21cmGAN paper, generating and saving samples, as well as a basic ABC rejection sampling algorithm can be found in `model_test.py`.

# Results
* Random samples for some parameter vectors:
<p align="center">
  <img width="572" height="334" src="https://github.com/FloList/21cmGAN/blob/master/pngs/random_samples_final.png">
</p>

* Resulting global 21cm signals:
 <p align="center">
  <img width="594" height="309" src="https://github.com/FloList/21cmGAN/blob/master/pngs/average_final.png">
 </p>

* Interpolating in parameter space:
<p align="center">
  <img width="373" height="714" src="https://github.com/FloList/21cmGAN/blob/master/pngs/interpolation.png">
</p>
