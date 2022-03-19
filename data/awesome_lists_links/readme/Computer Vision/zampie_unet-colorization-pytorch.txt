# unet_colorization_pytorch

A auto colorization application made by U-Net.
About the paper:
https://arxiv.org/abs/1505.04597  

For runing this program, you need:  
Pytorch: 1.1.0 or newer  
matplotlib


Result:

Illustrations  -->  Edges  -->  Generated

.<img src="https://github.com/zampie/unet_colorization_pytorch/blob/master/examples/199999_data_B.jpg" width="270"/>
.<img src="https://github.com/zampie/unet_colorization_pytorch/blob/master/examples/199999_data_A.jpg" width="270"/>
.<img src="https://github.com/zampie/unet_colorization_pytorch/blob/master/examples/199999_fake_B.jpg" width="270"/>


Usage:

Train:  
Run train.py, edit the file to change parameters.

Test:  
test.py, for test the whole test set  
test_costom.py, test one image  
test_folder.py, test a folder  

Get the pretrained model: 
https://drive.google.com/file/d/1vI4ZaK3ZnZtX3aah2E7iHwlnBJ59RYAS/view?usp=sharing


About the dataset:  
Illustrations are crawled from https://konachan.net/.  
Edges are extractd by a laplacian filter.
You can download the dataset from:
https://drive.google.com/open?id=1elHLWSpr-T2aSpAttYYMbDnj7sj5vs-Q

Examples:  
![image](https://github.com/zampie/unet_colorization_pytorch/blob/master/examples/Data_A.png) 
![image](https://github.com/zampie/unet_colorization_pytorch/blob/master/examples/Data_B.png) 


Some codes are from: https://github.com/milesial/Pytorch-UNet
