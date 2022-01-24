# Segmentor
- This is an implementation of DeepLab v3 (Chen et al., 2017 https://arxiv.org/abs/1706.05587) on TensorFlow 2.0 for the semantic segmentation problem on the Pascal VOC 2012 dataset (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- It achieved a 67% mIoU on the "val" set of Pascal VOC dataset
- A sample of the predictions is shown below:
![alt text](https://github.com/fjzs/Segmentor/blob/main/Segmentor%20samples.jpg)

# Who are we?
- Adi Nugraha (https://github.com/adinrh)
- Jarmila Ruzicka (https://github.com/Ruzick)
- Francisco Zenteno Smith (https://github.com/fjzs)

# Purpose:
- Implement and train a DeepLab v3 model from scratch
- Use an optimized training pipeline, particularly with the API from TensorFlow 2.0 (https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
- Compare this implementation to the pre-trained models available on TensorFlow Hub (https://tfhub.dev/s?dataset=pascal-voc-2012&module-type=image-segmentation)
- Deploy the application on AWS to make inferences on any .jpg image
- Analyze ethical considerations in the dataset

# How to install the project
- For deploying the model on AWS you would need an AWS account, nonetheless, it's pretty easy to train the model on a single noteboook and make predictions
- The specific instructions are detailed on the notebook (https://github.com/fjzs/Segmentor/blob/main/notebooks/Segmentor%20DeepLabV3%20v04.ipynb)

# Licence
MIT (https://github.com/fjzs/Segmentor/blob/main/LICENSE)
