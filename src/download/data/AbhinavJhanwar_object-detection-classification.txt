I have implemented following object/person detection models here namely-
1) ssd

# 1. SSD
* **Paper**- https://arxiv.org/pdf/1512.02325.pdf
* **Repo**- https://github.com/pierluigiferrari/ssd_keras
* **Methodology/technique**- ssd (single shot detection): dividing whole image into segments of fixed size then using various size of bounding boxes (6 per feature cell) and pass it through the network to classify. Here the difference is that this technique is applied not just on last conv layer but at multiple levels to handle scaling and then in the end nms (non maximum suppression) is used to get final bounding box of objects
* **Architecture/base network**- pretrained vgg16 layers + few custom conv layers and classificatin layers from various layers
* **Dataset trained on**- PASCAL VOC and COCO datasets
* **Input image size**- 300x300, 512x512
* **Optimizer**- sgd in original implementation and adam in the current implementation
* **Anchor boxes**- 4/6 default boxes per cell
* **loss function**- confidence_loss (logloss) + alpha*location_loss (smooth L1 loss)
