# Single Shot MultiBox Detector implemented by Keras


## Introduction

SSD(Single Shot MultiBox Detector) is a state-of-art object detection algorithm, brought by Wei Liu and other wonderful guys, see [SSD: Single Shot MultiBox Detector @ arxiv](https://arxiv.org/abs/1512.02325), recommended to read for better understanding.

Also, SSD currently performs good at PASCAL VOC Challenge, see [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=3](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=3)

## SSD Architecture 
Below find the architectural defference between Yolo and SSD. 
![title](https://github.com/nirajdevpandey/Object-detection-and-localization-using-SSD-/blob/master/Assets/Yolo_vs_SSD.png)

 

## Guides

The code structures looks like below:

```
1. Aseets - Prior boxes  (prior_boxes_ssd300.pkl is the model pre-defined static prior boxes)
2. Data-set - VOC-2007
3. SSD Model - The training and the test scripts 

  - ssd_v2.py # main model architecture using Keras
	- ssd_layers.py # Normalize and PriorBox defenition
	- ssd_training.py # MultiboxLoss Definition
	- ssd_utils.py # Utilities including encode,decode,assign_boxes
  
4.  data-generator  # customrized generator, which return proper training data structure
				            # including image and assigned boxes(similar to input boxex)
  - get_data_from_XML.py # parse Annotations of PASCAL VOC, helper of generator
  
  ```
## Walk-through

The multibox loss is consist of `L1 smooth loss` and `softmax` loss. Let's see how they llok like 

`Arguments`
    y_true: Ground truth bounding boxes,
	tensor of shape (?, num_boxes, 4).
    y_pred: Predicted bounding boxes,
	tensor of shape (?, num_boxes, 4).
	
`Returns`
    l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).
    
`References` - https://arxiv.org/abs/1504.08083

```python
def _l1_smooth_loss(self, y_true, y_pred):
	abs_loss = tf.abs(y_true - y_pred)
	sq_loss = 0.5 * (y_true - y_pred)**2
	l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
	return tf.reduce_sum(l1_loss, -1)
```
Now let's walk through the `softmax` loss

 `Arguments`
    y_true: Ground truth targets,
	tensor of shape (?, num_boxes, num_classes).
    y_pred: Predicted logits,
	tensor of shape (?, num_boxes, num_classes).
	
`Returns`
    softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
    

```python
def _softmax_loss(self, y_true, y_pred):
	y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
	softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
				      axis=-1)
	return softmax_loss
```
## Resources

dataset can be downloaded from [http://host.robots.ox.ac.uk/pascal/VOC/, use The VOC2007 Challenge in this example

Weights can be downloaded at [https://drive.google.com/file/d/0B5o_TPhUdyJWWEl5WG1lcUxCZzQ/view?usp=sharing](https://drive.google.com/file/d/0B5o_TPhUdyJWWEl5WG1lcUxCZzQ/view?usp=sharing)


## Hint

The folder called Data-set has just 5-10 samples of the VOC-2007. to train your own model please download the entire data set by clicking on the link above. Thanks 

## References

My work is just playing with this fantastic algorithm, and see the detection result of my own. Many many thanks goes to the author of the SSD paper
