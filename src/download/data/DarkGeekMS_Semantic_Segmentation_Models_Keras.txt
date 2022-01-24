# Semantic Segmentation Networks Keras Implementation

### Brief Description 

These are implementations for some neural network architectures used for semantic segmentation using the deep learning framework "Keras".

The Network Architectures Included:
1) Unet Architecture: https://arxiv.org/abs/1505.04597
2) Modified Unet with Resnet and VGG as Encoder.
3) Segnet Architecture (Resnet Encoder): https://arxiv.org/abs/1511.00561
4) Modified Segnet with Resnet and VGG as Encoder.
5) DeepLabv3: https://arxiv.org/abs/1706.05587


### Utilized Technologies and Frameworks

- Keras (Tensorflow Backend).
- Basic Python Data Manipulation Packages (Matplotlib,....etc).
- SciKit Learn.


### Repository Structure

**1) Model Files:**
- 5 files for the implementations of the previously mentioned architectures.

**2) Utilities Files:**
- **util_funcs.py**: contains some helper functions and the metric function.
- **preprocessing.py:** contains the code for preprocessing data before training.

**3) train.py:**
- contains training setup for differect models.

**3) test.py:**
- contains test setup.


### How to Use it 

- First, Clone the repository.
- To run model training: \
  -- First, you need to edit "train.py" file to set some variables indicated inside. \
  -- Then, run "train.py".
- To run inference: \
  -- First, you need to edit "test.py" file to set some variables indicated inside. \
  -- Then, run "test.py".
  
  
  
  Thanks a lot ^_^

