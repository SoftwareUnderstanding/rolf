# Prostate_Cancer_Grade_Assessment
Model for detecting prostate cancer on high resolution images of prostate tissue samples.
The goal is to accurately predict the ISUP grade (a grade for the severity of the cancer) of the tissue in question.

Prostate_Cancer_Grade_Classification.ipynb contains all code for image preprocessing, dataset creation, transfer learning model creation, and training.

This task is challenging for two reasons:
1) The images are huge, often times exceeding 100 million pixels, and downsampling reduces accuracy.
2) The images vary in height and width.
My strategy was to break the images into tiles and select only the 25 tiles that display the most tissue.
I used the 25 tiles as a batch of inputs and fed them into a modified version of EfficientNetB7 (see https://arxiv.org/abs/1905.11946).
I then concatenated the outputs and fed the combined array to a dense layer.
