# Gated_PixelCNN
*a Tensorflow implementation of the paper 《6527-conditional-image-generation-with-pixelcnn-decoders》*
https://arxiv.org/abs/1606.05328


**实验室大作业，通过已知的像素值，预测未知的像素，这里对于已知像素范围有明确界定**

1. 数据集MNIST(计算资源充足的可选择其他复杂数据集)

2. 按照逐行扫描的顺序预测后续像素

***对文章里提出的mask做了小的改变***

## Result
![Ground Truth](https://raw.githubusercontent.com/SJHNJU/Gated_PixelCNN/master/fig/1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)



![occulsion](https://raw.githubusercontent.com/SJHNJU/Gated_PixelCNN/master/fig/2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)



![output](https://raw.githubusercontent.com/SJHNJU/Gated_PixelCNN/master/fig/3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)
