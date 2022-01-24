# MobileFaceNet with Arc-Face Loss

***

## Requirments
```
numpy 1.15.1
keras 2.2.4
tensorflow-gpu 1.9.0
opencv-python 3.4.3.18
```
***

## Training
```
python train_MobileFaceNet.py train_directory data_directory pairs_filename  <--optional_arguments>
```

### Arguments
* Data Generator
* Model
* Training
* Eveluattion

#### Data Generator
* You can define your own data generator in ***src/data_generators.py***
<table width="600" border="4">
    <tr>
        <td><b>Argument</b></td>
        <td><b>Description</b></td>
        <td><b>type</b></td>
        <td><b>Default</b></td>
    </tr>
    <tr>
        <td>train_directory</td>
        <td>Training dataset directory</td>
        <td>str</td>
        <td></td>
    </tr>
    <tr>
        <td>valid_directory</td>
        <td>Validation dataset directory</td>
        <td>str</td>
        <td></td>
    </tr>
    <tr>
        <td>batch_size</td>
        <td>Batch size of generator.</td>
        <td>int</td>
        <td>200</td>
    </tr>
    <tr>
        <td>aug_freq</td>
        <td>Frequency of data augmentation.</td>
        <td>float</td>
        <td>0.5</td>
    </tr>
    <tr>
        <td>image_size</td>
        <td>Image size same as model input size.</td>
        <td>int</td>
        <td>112</td>
    </tr>
    <tr>
        <td>shuffle</td>
        <td>Shuffle on end of epoch.</td>
        <td>bool</td>
        <td>True</td>
    </tr>
</table>

#### Model
<table width="600" border="4">
    <tr>
        <td><b>Argument</b></td>
        <td><b>Description</b></td>
        <td><b>type</b></td>
        <td><b>Default</b></td>
    </tr>
    <tr>
        <td>expansion_ratio</td>
        <td>Expansion ratio of res_block.</td>
        <td>int</td>
        <td>6</td>
    </tr>
    <tr>
        <td>embedding_dim</td>
        <td>Embedding Dimension.</td>
        <td>int</td>
        <td>256</td>
    </tr>
    <tr>
        <td>loss_scale</td>
        <td>Scale parameter of arc-face loss.</td>
        <td>int</td>
        <td>64</td>
    </tr>
    <tr>
        <td>loss_margin</td>
        <td>Angular margin of arc-face loss.</td>
        <td>float</td>
        <td>0.5</td>
    </tr>
</table>

#### Training
<table width="600" border="4">
    <tr>
        <td><b>Argument</b></td>
        <td><b>Description</b></td>
        <td><b>type</b></td>
        <td><b>Default</b></td>
    </tr>
    <tr>
        <td>pretrained_model</td>
        <td>Pre-trained model filename.</td>
        <td>str</td>
        <td>None</td>
    </tr>
    <tr>
        <td>save_model_directory</td>
        <td>Directory to save model</td>
        <td>str</td>
        <td>weights/</td>
    </tr>
    <tr>
        <td>checkpoint_epochs</td>
        <td>Save checkpoint every n epochs.</td>
        <td>int</td>
        <td>5</td>
    </tr>
    <tr>
        <td>epochs</td>
        <td>Max number of training epochs</td>
        <td>int</td>
        <td>300</td>
    </tr>
    <tr>
        <td>valid_split_ratio</td>
        <td>Split ratio of validation set</td>
        <td>float</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>evaluate_epochs</td>
        <td>Evaluate model every n epochs</td>
        <td>int</td>
        <td>5</td>
    </tr>
</table>

#### Evaluation
<table width="600" border="4">
    <tr>
        <td><b>Argument</b></td>
        <td><b>Description</b></td>
        <td><b>type</b></td>
        <td><b>Default</b></td>
    </tr>
    <tr>
        <td>data_directory</td>
        <td>Evaluation dataset directory.</td>
        <td>str</td>
        <td></td>
    </tr>
    <tr>
        <td>pairs_filename</td>
        <td>Pairs file name</td>
        <td>str</td>
        <td></td>
    </tr>
    <tr>
        <td>sample_type</td>
        <td>
            <p>Sample type of the task.</p> 
            <p>0 : balance pos/neg</p>
            <p>1 : sample by person and img per person.</p>
        </td>
        <td><p>0</p><p>or</p><p>1</p></td>
        <td>0</td>
    </tr>
    <tr>
        <td>repeat_times</td>
        <td>Repeat times of generation, this argument only be used when sample type is 0.</td>
        <td>int</td>
        <td>10</td>
    </tr>
    <tr>
        <td>num_person</td>
        <td>Number of person to sample, this argument only be used when sample type is 1.</td>
        <td>int</td>
        <td>10</td>
    </tr>
    <tr>
        <td>num_sample</td>
        <td>Number of sample per person, this argument only be used when sample type is 1.</td>
        <td>int</td>
        <td>20</td>
    </tr>
    <tr>
        <td>far_target</td>
        <td>Target FAR(False Accept Rate)</td>
        <td>float</td>
        <td>1e-2</td>
    </tr>
</table>

#### GPU
<table width="600" border="4">
    <tr>
        <td><b>Argument</b></td>
        <td><b>Description</b></td>
        <td><b>type</b></td>
        <td><b>Default</b></td>
    </tr>
    <tr>
        <td>gpu</td>
        <td>Specify a GPU.</td>
        <td>str</td>
        <td>'1'</td>
    </tr>
</table>

***
## infer example
```python
import cv2 
import matplotlib.pyplot as plt
from src.build_model import ArcFaceLossLayer, dummy_loss
from src.feature_extractor import FeatureExtractor

# load model
model_path = 'weights/weights.h5'
fe = FeatureExtractor(model_path, num_classes=24)

# load image and resize to model input size 
path = 'images.jpg'
img = cv2.imread(path)[:, :, ::-1]
img_resize = cv2.resize(img, (112, 112))

# infer 
emb = fe.infer(img_resize)
emb.shape
>> (1, 512)
```
### Reference
ArcFace : https://arxiv.org/abs/1801.07698\
MobileFaceNet : https://arxiv.org/abs/1804.07573