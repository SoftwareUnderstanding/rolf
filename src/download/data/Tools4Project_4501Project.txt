# 4501Project：Paint Style Transformation
### Section1
### Group Name: PythonGroup
### Group Members: Tao Li, Jianxing Wan, Jiali Sun, Xiaojian Fan
<br>

## Main Objective
The main objective of this project is to enable users to transfer the style of their images, either an uploaded image or an image captured by their computer built-in camera when they click on the button `upload via camera` into any style they prefer, for example, it can provide the users with the magic capacity to paint like Vincent van Gogh!<br> 
<br>
A GUI is built to make it more easily to be operated by the users, especially the layman without the necessity to understand the code. Additionally, the working process and the final result can be more observable.<br>
<br>

## Convolutional Neural Network (CNN)
CNN is a technique introduced by Gatys et al. (2015) in their paper 'Image Style Transfer Using Convolutional Neural Network'. The algotithm allows us to separate and recombine the image content and style of natural images. In another words, it is capable of extracting content information from a image and then extract style information from a famous artwork, then combine them together to produce a new image that possesses the content of the photo and the appearance of a given artwork. The key idea backed up this algorithm is to optimise an image with the objective of matching desired CNN feature distributions, which involves both the photo's content information and the artwork's style information. The team realizes the function that image style transfer mainly by cnn_paint_style function from the website about "Painting Like Van Gogh with Convolutional Neural Networks"(2016).<br>
<br>
      
## GUI
The main interface is shown as below:<br>
<br>
![](https://github.com/Tools4Project/4501Project/blob/master/homepage.png)  <br>
<br>
As can be seen, there are three buttons on the main interface: <br>
<br>
`Upload Art`, when this button is clicked, it allows user to select an image (could be the famous artwork or any other images that the user want to extract the image's style) from their own computer, which if satisfies the format requirement, will be displayed on the left-hand-side of the interface. One important thing to be notified here is that only the 'jpg','jpeg','png','bmp' type of file can be uploaded, if the user try to upload one file other than these specified four or no file is selected, there will be a error message box popping up with the massage shown as "Please try to upload a 'jpg','jpeg','png','bmp' file.". <br>
<br>
`Upload your picture`, the button in the middle, when this one is clicked, there will be a pop up window with two other buttons indicating the two ways that the user can choose between to upload one picture that the user want to extract content information from: `Upload via local`: choose one from users' computer, `Upload via camera`: open computer's camera to take a photo. The user can press "c" to take a photo and press "q" to exit the camera interface. After one picture is seccessfully uploaded, it will be shown in the middle of the interface.<br>
<br>
After two pictures both have been uploaded, the user can finally click `process` button, a window will pop up, showing the users with ten intermediate points in the convolutional process and finally end up with the final result displayed on the right-hand side of the main interface. The window showing the convolutional process will be closed automatically. In this way, in addition to the final result, users can also see the convolutional process, which make it more transparent and interesting.<br>
<br>
The final result window is as the following:<br>
<br>
![](https://github.com/Tools4Project/4501Project/blob/master/result.jpeg)  
<br>


## Run Instructions
Download [imagenet-vgg-verydeep-19](http://www.vlfeat.org/matconvnet/pretrained/) into the directory (main directory) in which you will implement main functional files. Then, use the bash command `cd` to the main directory, `git init` initialize the main directory and produce git environment. Meanwhile, under the directory, use `git clone git@github.com:Tools4Project/4501Project.git` command to download the main functional files. Following this, use the code `python3.6 gui_main_page.py` to run the project, then the main gui will appear. Detailed operations of GUI have been shown above in 'GUI' section.
<br>









  
  
## References
1. Painting Like Van Gogh with Convolutional Neural Networks (Nov,2016).
   Retrieved from http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks

2. L. A. Gatys, A. S. Ecker, and M. Bethge (Aug. 2015), “A neural algorithm of artistic style,” ArXiv e-prints.
   Retrieved from: http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

3. Simonyan, Karen, Zisserman, & Andrew. (2015, April 10). Very Deep Convolutional Networks for Large-Scale Image Recognition. 
   Retrieved from https://arxiv.org/abs/1409.1556
