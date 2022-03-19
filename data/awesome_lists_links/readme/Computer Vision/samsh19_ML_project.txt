# ML Project: image segmentation with style transfer

This is a project for NYU intro to ML. The goal of this project is to combine the image segmentation, style transfer techniques to **create a partial style transfer image to a specific objective** (Note that for the segmentation, we only specify one object here).

This repository **didn't change much on the original code** but **provide the new display of the image outcome**.

### Image segmentation (Instant segmentation):
For the image segmentation, the **Mask RCNN** has been adapted here (https://arxiv.org/pdf/1703.06870.pdf). The Mask RCNN evolved from the Faster RCNN (RCNN -> Fast RCNN -> Faster RCNN -> Mask RCNN). Due to the one object detection scenario and followed from the sample code, I only use two classes (object and background) here. Note that for the multiple target distinguishment, just set the number of class to the required number (This project only supports two classes segmentation). 

### Style transfer:
For the style transfer, the **original version of style transfer** has been used here (https://arxiv.org/pdf/1508.06576.pdf). Instead of the style transfer, I think that it is more closely to the image synthesis. The model tunning the VGG16 and adding two different kinds of the loss function, content loss function and style loss function, then processed the optimization (gradient) with the content image and the style image. Note that the technique has an improved version from Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://arxiv.org/pdf/1603.08155.pdf) for better resolution and a more efficient way of predicting.

### Result:
<p align = 'center'>
<img src = 'https://github.com/samsh19/ML_project/blob/main/data/compare_images/polor_bear_japan_paint_wave_compare.png?raw=true'>
</p>
From the left to right, the content image, style image, style transferred image, segmentation of the content image, and the outcome

### Experinment:
To get the result, the steps have been listed below:
>1. Train the image segmentation model with `segmentation_modeling.ipynb`. The model will be saved in the `intermediate`. (Note that the training set is from https://www.cis.upenn.edu/~jshi/ped_html/)<br>
>2. Specific the content image in `data/content_images` and the style image in `data/style_images` for further usage in `style_transfer_base.ipynb`. We will generate the style transferred image and saved in `data/transfer_images`.<br>
>3. Load the segmentation model and the transferred image in `style_transfer_segmentation.ipynb`. Set the **boundary rate** to decide the **edge resolution**.

### Further improvement:
For the improvement, I will try on the Fast Style Transfer from the improved version to reduce the predicting time (each image will synthesis re-train each time for predicting). Since either the segmentation and style transfer technique all have the real-time processed version. I hope this project can generate a real-time processor for a similar technique in the future.

### Reference:
All the technique code is based on the PyTorch official tutorial document.
>https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html<br>
>https://pytorch.org/tutorials/advanced/neural_style_tutorial.html<br>

All images refer from the google search.