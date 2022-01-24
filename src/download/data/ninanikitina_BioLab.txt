# PROJECT NAME
**Recognition of BioImages based on Machine Learning algorithms**


# DESCRIPTION

This project solves the problem of detecting stress fibers (parallel actin filaments) located above and below nuclei within a single cell using a confocal microscope image of this cell.
![alt text](https://github.com/ninanikitina/BioLab/blob/master/readme_pic/research_project.png?raw=true)

There are two parts of the project:
1) Test part - detection volume of all nuclei in the picture with multiple nuclei. UNet model is used to detect countors of each nucleus on a picture, and the Alexnet model was used to detect real vs. reflected nucleus images on each layer.
UNet model was trained from scratch with 300 images of 30 nuclei on different slices (no data augmentation).
The result was compared with the result produced by the Imaris program. Test part overview and results: https://github.com/ninanikitina/BioLab/blob/master/readme_pic/Presentarion_12.18.2020.pdf
2) Main part - detecting stress fibers. Slices of the 3D image of a single nucleus are converted from XY axis to ZY axis, and a UNet model is used to detect countors of each fiber on each slice.  
The UNet model was trained from scratch with 40 images of different ZY slices of one nucleus (no data augmentation image).



# CREDITS

We used Pytorch-UNet cloned from https://github.com/milesial/Pytorch-UNet by @milesial
*Note from @milesial: Use Python 3.6 or newer*
*Unet usage section was copied from the README file from this repository*

# USAGE

## Drivers
- predict_nuclei_volumes - run prediction of nuclei volume on the image with multiple nucleus (Test part)

## Multiple Nuclei Utils - Test part
- czi_reader 
   Reads czi files and save jpg images from two different channels separately
   NOTE: initial jpg is 16 bits, and this script converts it to 8 bits by using hardcoded normalization.
   The user should decide based on images preview how he would like to normalize an image.
- cut_nuclei
   Cuts a big image into a bunch of out 512"512 (hard codded size) images form with nuclei in the center and
   creates a mask based on contours
- layers_to_3D
    Creates tif image of 3D visualization of detected nuclei
- reconstruct_layers
    Combine small mask images 512"512 into a big image
- test_normalization
    Test program to compare normalization for different thresholds
- volume_estimation
    Calculates the volume of all nuclei based on big reconstructed images (reconstruct_layers)

## UNet

### Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`


### Prediction

After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 1

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
