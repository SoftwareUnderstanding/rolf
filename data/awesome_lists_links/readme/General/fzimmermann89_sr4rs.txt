# sr4rs -- https://github.com/fzimmermann89/sr4rs/
Using SRResNet on UCMerced Landuse Dataset.
Limited myself to just the SRResNet without GAN, tried learning on the dataset, using prelearned weights from https://github.com/sgrvinod and finally using those weights as a starting point to learn on the dataset (transfer).

## Model
SRResNet (https://arxiv.org/abs/1609.04802), as implemented at https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

## Dataset
http://weegee.vision.ucmerced.edu/datasets/landuse.html

"[...] USGS National Map Urban Area Imagery collection for various urban areas around the country."

Resolution of 1px = 1ft.

21 classes, each 100  images, *mostly* 256x256 images

## Preprocessing
For downsampling, I chose a 2px Gaussian Blur followed by an 1/4-scaling using nearest neighbour as this seemed a good trade of between ease-of-implementation, staying close to the method suggested in the paper and a 'physically plausible' resolution degregation.

Training was done using randomly to 96x96px cropped and randomly flipped images as ground truth and downscaled to 24x24px images as input.

## Training
### Parameters

|Parameter     | Value|
---------------| -----| 
| batchsize    | 64   | 
| optimizer    | ADAM | 
| learning rate| 1e-4 | 
| iterations   | 1e5  | 

### Progress
#### Training on UCMerced

![](images/loss_landuse.png)

#### Using prelearned weights and training on UCMerced

![](images/loss_transfer.png)

## Results
Image Quality assesment using MSE, PSNR and SSIM (https://en.wikipedia.org/wiki/Structural_similarity) on the 210 images using the original 256x256px ground truth and downsampled to 64x64px as input.

|          | MSE     | PSNR     | SSIM      |
|----------|---------|----------|-----------|
| Bicubic  | 420±460 | 23.8±4.4 | 0.63±0.12 |
| UCMerced | 180±220 | 27.6±4.2 | 0.79±0.09 |
| Sgrvinod | 320±360 | 25.0±4.3 | 0.69±0.12 |
| Transfer | 170±210 | 27.9±4.3 | 0.80±0.09 |

### Histograms of quality metrics
#### Training on UCMerced
![](images/hist_landuse.png)
#### Using sgrvinod's prelearned weights
![](images/hist_sgrvinod.png)

#### Using prelearned weights and training on UCMerced
![](images/hist_transfer.png)

### Example images
Low resolution is downscaled (see preprocessing) Ground Truth image, for plotting upscaled with nearest neighbour without any other interpolation. Comparision between SRResNet and Bicubic upscaling.
#### Training on UCMerced
![](images/landuse/img0.png)
![](images/landuse/img1.png)
![](images/landuse/img2.png)
![](images/landuse/img8.png)
![](images/landuse/img4.png)

#### Using sgrvinod's prelearned weights
![](images/sgrvinod/img0.png)
![](images/sgrvinod/img1.png)
![](images/sgrvinod/img2.png)
![](images/sgrvinod/img8.png)
![](images/sgrvinod/img4.png)

#### Using prelearned weights and training on UCMerced
![](images/transfer/img0.png)
![](images/transfer/img1.png)
![](images/transfer/img2.png)
![](images/transfer/img8.png)
![](images/transfer/img4.png)




### Discussion
Limitations:
 - I did not spend much time tuning hyperparameters
 - Unfortunatly, in the validation phase my PSNR calculation was wrong (the images were scaled -1..1, the wikipedia definition works if the values start at 0)

Should compare images with unsharp masking for subjective judging, as the bicubic rescaled images are generally less sharp and this dominates the subjective image quality.

Used image quality metrics -- *NOT* resolution in the, e.g modulation transfer function etc.
For using this scheme as a preprocessing for *quantitive* image interpretation, MTF resolution might be more important

### Project Structure
#### src
 - `ds.py` - Contains reader for directory of tif files as used in the UCMerced Dataset and a Dataset-wrapper which downsamples the input dataset
 - `model.py` - SRResNet Model
 - `util.py` - small utilities
#### Notebooks
 - `dataset.ipynb` - cursory glance at the dataset, with regards to image dimensions and histogram
 - `train.ipynb` - training, creates weights and log file
 - `results.ipynb` - plot results, creates all the images used in the Readme
 
#### Results
 - `weights` contains trained weights as state_dict's
 - `images` contains all images used for this readme
