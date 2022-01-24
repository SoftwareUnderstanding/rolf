Neural Style Transfer with TensorFlow

##  ___**Overview**___
1. The code for Neural Style Transfer is in the **NST_Tensorflow** Jupyter Notebook. The Python file **nst_setup**, which builds up the Neural Network, is automatically called by the Notebook
2. The provided content and style images reside in the **challenge** folder
3. All 10 images generated while the model was running (1 after every 10th iteration, 100 iterations in total) are in the **Generated_Images** folder. The *final output image* matching both content and style is named **iter100**, and is in the same folder

##  ___**Model used**___
VGG 19 pre-trained on ImageNet data, max-pooling replaced with average-pooling, last 3 FC layers removed 

##  ___**Conclusion**___
Neural Style transfer was successfully implemented, and this can be seen by going through the generated images in sequence (**iter10** to **iter100**). Only 100 update iterations were carried out due to the limited computing power of my CPU. Still, the final image matches both the content of the Japanese garden and the style of the Picasso portrait. 

##  ___**Papers**___
* VGGNet: https://arxiv.org/abs/1409.1556
* Neural Style Transfer: https://arxiv.org/abs/1508.06576

##  ___**Modules used**___
* PIL: https://pillow.readthedocs.io/en/stable/
* matplotlib: https://matplotlib.org/api/pyplot_api.html
* numpy: https://numpy.org/
* tensorflow: https://www.tensorflow.org/versions/r1.14/api_docs/python (tf version 1.14 used)
* keras.applications: https://keras.io/applications/
