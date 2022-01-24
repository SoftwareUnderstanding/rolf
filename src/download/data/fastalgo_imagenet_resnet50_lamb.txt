How to reproduce?

Just hit ./run.sh

#################################################################################################

Please check log.out for the training log.

#################################################################################################

This is an implementation of LAMB optimizer by TensorFlow for ImageNet/ResNet-50 training.

Large Batch Optimization for Deep Learning: Training BERT in 76 minutes

https://arxiv.org/pdf/1904.00962.pdf

Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh

#################################################################################################

This implementation can get 76.3% accuracy for ImageNet/ResNet-50 training in just 3519 iterations (batch size = 32K).

State-of-the-art optimizer like Adam fails to achieve this level of accuracy for large-batch training. 

The authors significantly tuned the hyper-parameters of Adam in https://arxiv.org/pdf/1904.00962.pdf

#################################################################################################

We use 128 v3 TPU chips in this experiment. Because of the distributed batch normalization, the accuracy will be higher if you increase the number of chips to 256 or 512.

#################################################################################################
