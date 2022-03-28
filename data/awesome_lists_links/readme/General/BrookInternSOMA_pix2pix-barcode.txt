# pix2pix-tensorflow

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

[Interactive Demo](https://affinelayer.com/pixsrv/)

Tensorflow implementation of pix2pix.  Learns a mapping from input images to output images, like these examples from the original paper:

<img src="docs/examples.jpg" width="900px"/>

This port is based directly on the torch implementation, and not on an existing Tensorflow implementation.  It is meant to be a faithful implementation of the original work and so does not add anything.  The processing speed on a GPU with cuDNN was equivalent to the Torch implementation in testing.

## Setup

### Prerequisites
- Tensorflow 1.12.0

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started

```sh
# clone this repo
git clone https://github.com/BrookInternSOMA/pix2pix-tensorflow.git
cd pix2pix-tensorflow
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python pix2pix.py \
  --mode train \
  --output_dir barcode_train \
  --max_epochs 1000 \
  --input_dir barcode/train \
  --which_direction BtoA

# load checkpoint
  --checkpoint ./barcode_train

# test the model
python pix2pix.py \
  --mode test \
  --output_dir barcode_test \
  --input_dir barcode/val \
  --checkpoint barcode_train
```

### Creating your own dataset

```sh
# Resize source images
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --output_dir photos/resized

python tools/process.py \
  --input_dir photos/blank \
  --operation resize \
  --output_dir photos/blank
  
# Create images with blank centers
python tools/process.py \
  --input_dir photos/resized \
  --operation blank \
  --output_dir photos/blank
  
# Combine resized images with blanked images
python tools/process.py \
  --input_dir photos/resized \
  --b_dir photos/blank \
  --operation combine \
  --output_dir photos/combined
  
# Split into train/val set
python tools/split.py \
  --dir photos/combined
```

The folder `photos/combined` will now have `train` and `val` subfolders that you can use for training and testing.

#### Creating image pairs from existing images

If you have two directories `a` and `b`, with corresponding images (same name, same dimensions, different data) you can combine them with `process.py`:

```sh
python tools/process.py \
  --input_dir a \
  --b_dir b \
  --operation combine \
  --output_dir c
```

This puts the images in a side-by-side combined image that `pix2pix.py` expects.

#### Colorization

For colorization, your images should ideally all be the same aspect ratio.  You can resize and crop them with the resize command:
```sh
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --output_dir photos/resized
```

No other processing is required, the colorization mode (see Training section below) uses single images instead of image pairs.

## Training

### Image Pairs

For normal training with image pairs, you need to specify which directory contains the training images, and which direction to train on.  The direction options are `AtoB` or `BtoA`
```sh
python pix2pix.py \
  --mode train \
  --output_dir barcode_train \
  --max_epochs 200 \
  --input_dir barcode/train \
  --which_direction BtoA
```

### Tips

You can look at the loss and computation graph using tensorboard:
```sh
tensorboard --logdir=barcode_train
```

<img src="docs/tensorboard-scalar.png" width="250px"/>

If you wish to write in-progress pictures as the network is training, use `--display_freq 50`.  This will update `barcode_train/index.html` every 50 steps with the current training inputs and outputs.

## Testing

Testing is done with `--mode test`.  You should specify the checkpoint to use with `--checkpoint`, this should point to the `output_dir` that you created previously with `--mode train`:

```sh
python pix2pix.py \
  --mode test \
  --output_dir barcode_test \
  --input_dir barcode/val \
  --checkpoint barcode_train
```

The testing mode will load some of the configuration options from the checkpoint provided so you do not need to specify `which_direction` for instance.

The test run will output an HTML file at `barcode_test/index.html` that shows input/output/target image sets:

<img src="docs/test-index-html.png" width="300px"/>

## Exporting

Exporting is done with `--mode export`.  You should specify the export directory to use with `--model_dir`:

```sh
python pix2pix.py \
  --mode export \
  --output_dir your_export \
  --checkpoint your_checkpoint
```

You use this exporting model by below command 

```sh
python server/tools/process-local.py \
  --model_dir your_export_dir \
  --input_file your_input_image_filename \
  --output_file output_filename \
```

You can use this for many inputs

```sh
python server/tools/process-local-dir.py \
  --model_dir your_export_dir \
  --input_dir directory_containing_your_input_images \
  --output_dir output_directory \
```

## Citation
If you use this code for your research, please cite the paper this code is based on: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation Using Conditional Adversarial Networks</a>:

```
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}
```

## Reference
- [affinelayer/pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

## Acknowledgments
This is a port of [pix2pix](https://github.com/phillipi/pix2pix) from Torch to Tensorflow.  It also contains colorspace conversion code ported from Torch.  Thanks to the Tensorflow team for making such a quality library!  And special thanks to Phillip Isola for answering my questions about the pix2pix code.
