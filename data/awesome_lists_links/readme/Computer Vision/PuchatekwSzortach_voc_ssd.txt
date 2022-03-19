# voc_ssd
Single Shot MultiBox Detector (SSD) implementation for PASCAL VOC 2012 dataset.

While the general concepts of vanilla SSD algorithm are maintained, several important differences and additions to reference implementation are introduced:
- data analysis
- maximum theoretical network recall analysis
- only blocks 2 to 5 of VGG backbone are used for predictions
- configurable network architecture
- use of corner boxes instead of center boxes
- simpler offsets predictions loss
- network operates on original image resolution

A short description of each change follows.

### Data analysis
Original paper defined sizes and prediction layers for default boxes used in the network without any reference to sizes of objects it was trying to detect.
In this work we provide a script to analyze sizes and aspect ratios of objects in data set, which can be used to guide network architecture design. 

### Maximum theoretical network recall analysis
Given sizes and arrangement of default boxes in SSD, it's possible to compute what is the maximum theoretical recall for the network against a given dataset.
Knowing that ceiling helps to inform both training and architecture design decisions.

### Only blocks 2 to 5 of VGG backbone are used for predictions
From data analysis one can compute set of optimal default boxes and their placement for given dataset.
It turns out that for PASCAL VOC 2012 dataset using layers above block 5 is not necessary - all objects in the dataset
can be detected with boxes placed on earlier layers.  
Analysis also shows that over 70% of annotations require default boxes placed on block 2 and 3 for default boxes
to be able to match them.

### Configurable network architecture
Configuration file provided allows to control which of major VGG blocks outputs should be used to construct prediction
heads, as well as what should be sizes and aspect ratios of default boxes placed on them. 
No changes in code are necessary to adjust network to configuration optimal for a given dataset. 

### Use of corner boxes instead of center boxes
Original SSD network defines default boxes in `[center_x, center_y, width, height]` format.  
This work uses an alternative `[min_x, min_y, max_x, max_y]` format.
Both formats are interchangeable, but the latter is far more popular among computer vision frameworks and easier to work with.

### Simpler offsets predictions loss
Offsets losses are just square error losses scaled by boxes sizes, and computed in the same fashion for each box coordinate.

### Network operates on original image resolution
Original SSD scales images to 300x300 or 500x500 resolution.
This has several disadvantages, especially for VOC dataset:
- objects aspect ratios might be distorted - and the distortion factor varies across images 
- data analysis becomes more difficult, making finding optimal network configuration difficult as well
- for most VOC images above rescaling decreases image resolution, making small objects, so ones that are particularly hard to detect, even smaller

In this work we choose to train and predict on original image resolution, only adjusting image to a size factor of 32, which simplifies computations of default boxes coordinates.
This still allows predictions to run above 30 frames per second on GeForce GTX 1080 Ti.

### Provided scripts
Following scripts are provided in the `scripts` directory
- `data_analysis.py` - analyzes sizes and aspect ratios of annotations in dataset
- `model_analysis.py` - analyzes performance of trained model on a dataset
- `networks_configuration_analysis.py` - analyzes overlap between neighbouring default boxes defined by network configuration
- `networks_theoretical_bounds_analysis.py` - analyzes theoretical recall network with given configuration can achieve on given dataset, reports sizes and aspect ratios of annotations network can't detect
- `train.py` - trains network
- `visualize.py` - provides routines to visualize raw data, augmented data, predictions, etc

Location of data and model paths, training hyperparameters and other inputs for all scripts are controlled through configuration file parameter.
`config.yaml` provides a sample configuration.

### Using with PASCAL VOC 2012 dataset

Dataset is not included with this repository. Please download dataset from the [official webpage](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).
Once downloaded, adjust config.yaml so its relevant section points to path with data.

A few sample predictions on VOC 2012 dataset made with a trained model are shown below

#### Good prediction
![alt text](./images/good_prediction.png)  

#### Typical prediction - many objects are correctly detected, but a few are off
![alt text](./images/typical_prediction.png)  

#### Bad prediction
![alt text](./images/bad_prediction.png)  

### Using with other datasets

This project can be readily reused with different object detection datasets.
In most cases the only changes you would need to do are:
- implement a data loader - look at `net.data.VOCSamplesDataLoader` for reference
- adjust configuration file to load data from appropriate path

Of course I would then advise to use tools project provides to define optimal network configuration for your dataset, going through  
`data analysis -> network configuration adjustments -> theoretical network performance analysis loop -> training -> model performance analysis`  
loop.

