## Deep Inside Convolutional Networks
## Paper: Simonyan, K., Vedaldi, A., Zisserman, A. 2013. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.
## Link: https://arxiv.org/abs/1312.6034v2

reference:
* [0]https://arxiv.org/abs/1312.6034v2
* [1]https://arxiv.org/abs/1409.1556
* [2]https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
* [3]https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
* [4]https://github.com/ivanmontero/visualize-saliency/blob/master/saliency.py
* [5]https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
* [6]https://stackoverflow.com/questions/58544097/turning-off-softmax-in-tensorflow-models
* [7]https://farm1.static.flickr.com/104/304394105_ee88b931d8.jpg
* [8]https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/bird.jpg

This project try to implement this paper on **VGG16 model** using **keras** with **tensorflow backend**.
 
1. visualise first Conv Layers' filters
2. visualise activation map of a input image
3. visualise image that can maximise one class score
4. visualise saliency map of a input image

1, 2 and 4 have been successfully implemented. However, I cannot achieve 3 very well.

I use the formula(1) given in the paper, and use zero image plus ImageNet weights as input. I also calculate result with and without image preprocessing as well as deprocessing using zero-centred method with ImageNet weights. Nevertheless, the generated image is also undiscriminatable.

Therefore, I use methods mentioned in [3]. Besides, I create a zero image with e*20 noise per pixel, where e randomly are sample from np.random.random(). Then plus ImageNet weights. For depreprocessing, I use methods in [3].

You can find more details in *Visualising Image Classification Models and Saliency Maps.ipynb*.

### If you have any suggestions or questions, please create issue, Thank you!