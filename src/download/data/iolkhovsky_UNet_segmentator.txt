## UNet segmentator

This python project implements U-Net image segmentation model described in https://arxiv.org/abs/1505.04597 using torch framework. Source paper added to docs folder.

![Alt text](attachments/segmenter.gif?raw=true "Example of prediction by raw (fast trained on VOC2007) model")

### Dataset preparing
Use voc_index.py script to create index (binary with found sample paths and some service info) for VOC-like dataset. 

### Training
Use train.py script to train pre-trained or raw model. Use tensorboard to follow training process (loss curves, accuracy, example of predicted while validation samples) 

![Alt text](attachments/loss.png?raw=true "Loss curve")
![Alt text](attachments/valid.png?raw=true "Predicted and ground-truth maps")
![Alt text](attachments/valid.png?raw=true "Source images")

### Evaluation
Use deploy.py script to evaluate trained model with web-camera

### Testing
Use eval_model.ipynb to test accuracy for trained model


