# Stanford-cs230-final-project

This project uses a semantic segmentation approach to tackle the freeway lane detection problem. 

## Model
Our model is modified from a widely used semantic segmentation model called Segnet: https://arxiv.org/abs/1511.00561. Our implementation of Segnet is based on an existing keras implementation (https://github.com/divamgupta/image-segmentation-keras). The keras code loads pre-trained VGG-16 weights for Segnet's encoder. 

### Baseline model
We compare our model with Lanenet (https://arxiv.org/pdf/1802.05591v1.pdf), a state-of-the-art lane detection model that won 4th in TuSimple Lane Detection Challenge. We use an exisitng Tensorflow implementation of LaneNet (https://github.com/MaybeShewill-CV/lanenet-lane-detection). The repo contains pre-trained weights of LaneNet.
 

## Dataset
We use the TuSimple dataset: https://github.com/TuSimple/tusimple-benchmark, which contains 2858 images. And we split the data into train (80%), dev (10%), test (10%).

### pre-processing
use ./tools/process to pre-process our data. The script takes in the label of TuSimple Dataset (in Json format) and generates a binary image for the semantic segmentation. Usage as follows:

```
python tools/process.py --home_dir YOUR_HOME_DIR_CONTAIN_DATA
```

The home_dir is where you store all the clips of your dataset. The script saves its outputs in two separate folders. home_dir/original_image saves all original images, where home_dir/label_image saves all binary label images. The image names in original_image folder and label_image folder are identical. For example, an image in path home_dir/clips/0313-1/60/20.jpg will generate two images named clips_0313-1_60.png in both home_dir/original_image and home_dir/label_image.

### post-processing

To start post-processing, you need to be read three text files: train.txt, val.txt, test.txt, where each one of them contain file names for train, dev and test set respectively. Run 

```python tools/convert_binary_to_label.py --home_dir YOUR_HOME_DIR```

It will generate 3 jsons train_pred.txt, val_pred.txt and test_pred.txt. These txts contain the json labels converted from the binary predict images from our semantic segmentation model.

### evaluate
Run ```python tools/evaluate.py YOUR_PRED_JSON_TXT GROUND_TRUTH_JSON_TXT```

## DEMO
We randomly downloaded some freeway videos online (outside of our training/dev/text set) and ran our model on it. 
![](demo.gif)

You can find complete video here: https://youtu.be/py-JNsGZ6_g


We will start from the baseline mode and build our lane detection model. 
