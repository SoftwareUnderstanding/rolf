
# Annotation:

* **Framwork**

  Pytorch

* **Dataset**

  In this case, I wrote a script to divide VOC2012 dataset into three parts as train\validation\test,the proportion is 8:1:1.And each part hava a images directory and a tags.txt.The tags.txt record the image path and label.So you can use this file to batch load the input data.Paticularly,the label normally hava more than one tag refer to the Annonciation.xml which you can easily find in the VOC2012 dataset.(If necessary,email dssbxxpersonal@163.com for the script) 

* **Model**

  For generality,the offical code of ResNet model is too fussy.So i simpled a brief model only for ResNet-50.

  He Kaiming's paper：https://arxiv.org/abs/1512.03385  
  Official code：https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py

* **Loss and Acuuracy for K-labels problem**

  loss: built-in loss function torch.nn.MultiLabelSoftMarginLoss()  
  accuracy:self defined in code.torch.nn.BCELoss is more recommended.But actually there no difference.(personal opinion...)

* **Result**

  see out.txt (90.15% accuracy for only 20 epoch)  

more details email dssbxxpersonal@163.com
