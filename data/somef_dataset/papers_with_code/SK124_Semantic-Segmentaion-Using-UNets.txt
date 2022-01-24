### Pixel Classifier
### Image Segmentation using UNets ###
* Image Segmentation is classifying every pixel of input image to one of several classes which we assign initially,which means our model will output the same image but with every pixel classified into one of many categories like 'Air','Pedestrian',also our final image will be of similar dimensions of the input so we can't use normal architectures that we use for classification tasks beacause it downsamples our images into vectors but in our case we need a upsampling path to restore the images to its original dimensions.How do we achieve  this?
* While converting an image into a vector, we already learned the feature mapping of the image so why not use the same mapping to convert it again to image.This is the recipe behind UNet. Use the same feature maps that are used for contraction(downsampling) to expand a vector to a segmented image. This would preserve the structural integrity of the image which would reduce distortion enormously. 
Let’s understand the architecture more briefly.
## Architecture ##
![1_lvXoKMHoPJMKpKK7keZMEA](https://user-images.githubusercontent.com/47039231/75104203-37075a80-562c-11ea-8c80-f8af704e9b28.png)
* This architecture consists of three sections: The downsampling path, The bottleneck, and the upsampling path. The downsampling section is made of many contraction blocks. Each block takes an input applies two 3X3 convolution layers followed by a 2X2 max pooling. The number of kernels or feature maps after each block doubles so that architecture can learn the complex structures effectively. The bottommost layer mediates between the downsampling layer and the upsampling layer. It uses two 3X3 CNN layers followed by 2X2 up convolution layer.
* Importance lies in the upsampling path which differentiates it from other architectures because it consists of several expansion blocks. Each block passes the input to two 3X3 CNN layers followed by a 2X2 upsampling layer. Also after each block number of feature maps used by convolutional layer get half to maintain symmetry. However, every time the input is also get appended by feature maps of the corresponding contraction layer. This action would ensure that the features that are learned while contracting the image will be used to reconstruct it. The number of expansion(upsampling) blocks is as same as the number of contraction(downsampling) block. After that, the resultant mapping passes through another 3X3 CNN layer with the number of feature maps equal to the number of segments desired.



**Metric:**
* We are classifying each pixel into one of the classes. The idea is that even in segmentation every pixel have to lie in some category and we just need to make sure that they do. So we just converted a segmentation problem into a multiclass classification problem,you can also write your own variant of loss function but the basic principle (every pixel belongs to one of many categories) remains same.
* Loss function I used in my problem

        def accuracy(input,target):
         target = target.squeeze(1)
         mask = (target != void_code)
         return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


**Dataset**
* Im using a variant of CamVid dataset
* Original: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
* Variant:  https://github.com/alexgkendall/SegNet-Tutorial.git 
* Full Dataset has 32 classes however to reduce complexity im using the important ones which are 
* codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
    'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
* Labelled Input batch:
 * <img width="606" alt="batc" src="https://user-images.githubusercontent.com/47039231/75104816-5ce42d80-5633-11ea-9f25-113d4d336481.PNG">
 

**Results**  
* <img width="397" alt="table2" src="https://user-images.githubusercontent.com/47039231/75104650-b8adb700-5631-11ea-86f4-a57dbdc2eb88.PNG">
<img width="506" alt="Capture" src="https://user-images.githubusercontent.com/47039231/75104821-77b6a200-5633-11ea-9041-173b8309ffce.PNG">

* Do Checkout My Colab Notebook for the Complete Implemntation. https://colab.research.google.com/drive/17txWDxQ0y18v8oFErQfsqPypkaLotVJt

### Acknowldgements
* Used GPU offered by Google Colabaroty
* Do check out fast.ai course Lesson 7 (https://course.fast.ai/videos?lesson=7) 
* Library Used: fast.ai v2
* Paper  https://arxiv.org/pdf/1505.04597.pdf
* Additional Info. https://www.jeremyjordan.me/semantic-segmentation/





  
