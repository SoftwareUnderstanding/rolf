
#  AI/DL EfficientNet Fruits Detection 


## Introduction

This is an neural network web app visualizing the training of the network and testing accuracy ~ 99.5% accuracy. The neural network uses pretrained EfficientNet_B3 and then trained to classify images of fruits and vegetables. It is built using Pytorch framework using Python as primary language. App is built using Flask.
**© Toàn Vinh**

## Guideline Run APP

Make sure you have installed Python , Pytorch, Flask and other related packages.

First download all the folders and files

git clone https://github.com/toanbkmt/EfficientnetFruitDetect.git

Then open the command prompt (or powershell) and change the directory to the path where all the files are located.

cd EfficientnetFruitDetect

Now run the following commands -

python app.py

This will firstly download the models and then start the local web server.

now go to the local server something like this - http://127.0.0.1:5000/ and see the result and explore.

## Summary Table

|            |                   |       
| ---------- |-------------------|
| **Author**       | Trần Vĩnh Toàn- Foundation 8 - VTC AI|
| **Title**        | Computer Vision - EfficientNet For Fruits & Vegetables Detection App  |
| **Topics**       | Ứng dụng trong computer vision, sử dụng thuật toán chính là CNN|
| **Descriptions** | Input sẽ là tấm hình với các loại quả khác nhau và file labels-v2.txt chứa danh sách tên của 130 loại quả tương ứng. Dữ liệu dùng để train là dataset của 130 loại quả có kích thước (100px X 100px). Train toàn bộ dữ liệu này bằng cấu trúc mạng CNN  sử dụng model EfficientNet ( Chi tiết về model : https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet, Paper: https://arxiv.org/abs/1905.11946). khi train xong sẽ trả ra output là file trọng số ```weights```. Ta sẽ sử dụng trọng số ```weights``` đã train để predict name của các object trong hình|
| **Links**        | https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet , https://github.com/rwightman/pytorch-image-models|
| **Framework**    | PyTorch|
| **Pretrained Models**  | sử dụng weight đã được train sẵn [https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra-a5e2fbc7.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra-a5e2fbc7.pth) [https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth)|
| **Datasets**     |Mô hình được train với bộ dữ liệu 130 loại quả tại: https://www.kaggle.com/moltean/fruits/data|
| **Level of difficulty**| Normal +, có thể train lại với tập dữ liệu khác và model khác tốc độ tùy thuộc vào CPU & GPU & data input|

## Test Result  

This is result test by input image with multiple fruit and test try detection & effective of modell trained
![](https://i.imgur.com/qX9glL9.png)


## Final Poster

| **Title**| EfficientNet Fruits & Vegetables Detect|
| ---------- |-------------------|
| **Team**| Trần Vĩnh Toàn ( toanbk.dev@gmail.com)|
| **Predicting** | App had been built with wish make app can detect multiple fruits & vegetable & because I like topic about computer vision and deep leaning.  I built app use neural network with main algorithm CNN and using pretrain model EfficientNet B3, dataset of 130 Fruits & Vegetables, use framework PyTorch. **Result**: After train 50 epoch , Accuracy of validation data up to **99,6%**, App can detect correct **80%** name of fruit in image test data. Summarry: **input:** image with multiple fruits & vegetable, **output:** name of fruit & vegetable exist in image input.|  
| **Data**       | Dataset of 130 fruits & vegetables at https://www.kaggle.com/moltean/fruits/data , foreach image have size (100px X 100px) image type color.|
| **Features**   | input data have 194 feature, the raw input data (color, weight, location) vs. features you have derived (ex. ICA, Gaussian Kernel)? They're appropriate for this task because it can use these features to classify and identify objects that match a pattern.|
| **Models**     | use model **EfficientNet** .Detail basic math & fomulas of Model at  https://github.com/rwightman/gen-efficientnet-pytorch  |
| **Results**    | When I try train use model **EfficientNet_B3** --> result (Best validate Acc: **0.99453696**, Best Validation Loss: **0.02312439**. With model **EfficientNet_ES**   --> result (Best validate Acc: **0.99418451**, Best Validation Loss: **0.03929441**.|
| **Discussion** | In the process of making the project, I have some interesting things to share:  [-] I used EfficientNet_B3 model to train with existing dataset, I used optimize SGD function, I trained with 10 epoch, the Accuracy result on validation set reached 98%. And I tested the ability to identify the actual accuracy of about 90%.  [-] Then I want to improve the ability to identify more accurately, should continue to train 30 epoch with the optimization parameters as before, the Accuracy on the validation set reached 99%. And I started testing whether the model's detect capability worked, I hope it got better. But the actual result this time was worse than the previous detect error to 40%. I found out that my train model was **Overfitting**.|
| **Future**     | I think I will study to find ways to improve the accuracy and overcome the overfitting problems that my project has encountered. At the same time further develop the function of the app is able to label the names of each fruit directly in the resulting image|
|**References**  |[IEEE style](https://ctan.org/topic/bibtex-sty?lang=en) is fine|


