# Deep Into CNN

Contains material relevant to "Deep Into CNN" Project.

## Resources

### Week 1 : Regression( Skip if you are confident )

#### Readings
1. Local Setup (Use Conda : recommended)  
https://jupyter.readthedocs.io/en/latest/install/notebook-classic.html  
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation  
2. (Optional: Basic Python and libraries)  
https://duchesnay.github.io/pystatsml/index.html#scientific-python  
3. ( Optional : For those with very basic ml knowledge: Only 2.1-2.7)  
https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN
4. Linear Regression:  
 https://medium.com/analytics-vidhya/simple-linear-regression-with-example-using-numpy-e7b984f0d15e  
5. Logistic Regression:  
https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc  

#### Practice Material
Find in [NeuralNetIntro](W2-3/NeuralNetIntro/) : W2-3.

### Week 1-2: Neural Networks

#### Readings
1. This one is highly recommended:  
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi   
Some more material (bit extensive, so be careful):  
https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
2. Basic Backprop:  
 https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html 
3. Backprop (Mathematical Version):  
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
4. Softmax:  
https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
5. Pytorch(Skip the CNN part if you want for now):  
https://pytorch.org/tutorials/beginner/basics/intro.html
6. Optional guide:  
http://neuralnetworksanddeeplearning.com/chap1.html
7. Pytorch Autograd:  
https://www.youtube.com/watch?v=MswxJw-8PvE&list=PL-bzqKhHrboYIKgBwoqzl6-eyCHP3aBYs&index=4

#### Practice Material

Find in [PyTorch](W2-3/PyTorch) : W2-3.

### Hackathon 1
June 1 - June 30 :  
https://www.kaggle.com/c/tabular-playground-series-jun-2021

Find Sample Submission [Here](W2-3/Hackathon1.ipynb)
### Week 2-3: Convolutional Neural Networks

#### Readings

1. These will give you a good Intuition:  
https://www.youtube.com/watch?v=py5byOOHZM8   
and  
https://www.youtube.com/watch?v=BFdMrDOx_CM  
Also, do check this blog out:  
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 
2. Highly Recommended:    
https://cs231n.github.io/convolutional-networks/ 
3. (L2-L11 : Enough for Understanding Implementation Details):  
https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
4. Pytorch official guide (Try after doing [W3 exercises](W3)):  
https://pytorch.org/tutorials/beginner/basics/intro.html

#### Practice Material

Find in [W3 Folder](W3)

### Paper 1 Implementation
Choose Any 1 of following:
1. AlexNet:  
https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
2. VGG:  
https://arxiv.org/pdf/1409.1556v6.pdf
3. Inception(GoogLeNet)*:  
https://arxiv.org/pdf/1409.4842v1.pdf
4. Xception*:  
https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
5. ResNet*:  
https://arxiv.org/pdf/1512.03385v1.pdf  

(* Recommended)

#### Supplement Material
1. Inception Module:  
https://towardsdatascience.com/deep-learning-understand-the-inception-module-56146866e652
2. Separable Convolutions:  
https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
3. Implementation of Xception :
Use `groups` argument of conv2d for separating channels (i.e. for Depthwise Separable Convolution ):  
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

### Hackathon 2
Based on following Dataset:  
https://www.kaggle.com/gpiosenka/100-bird-species

Find Sample Submission in [W6 Folder](W6/Hackathon2_xception.ipynb)

### Week 4-5: Optimization

1. Visualizing MNIST (Casual Reading, Enjoy Animations):   
http://colah.github.io/posts/2014-10-Visualizing-MNIST/

2. Optimizers: Only Gradient Descent Variations, Adam and RMSProp:  
https://ruder.io/optimizing-gradient-descent/

3. SGD with Momentum(Mathematical,For future reference)  
https://distill.pub/2017/momentum/

4. Weight Initialization:  
https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

5. Batch Norm :  
https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338

6. Overfitting, Regularization, Hyper-parameter tuning :  
http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network%27s_hyper-parameters

7. Complete Reference(Videos):  
https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc

### Week 6: VAE and GANs

#### Readings
1. Deep Generative Modelling(Must Watch):  
https://www.youtube.com/watch?v=BUNl0To1IVw

2. AutoEncoders:  
https://www.jeremyjordan.me/autoencoders/

3. Variational AutoEncoders:  
https://www.jeremyjordan.me/variational-autoencoders/

4. GAN Intuition:  
https://www.youtube.com/watch?v=Sw9r8CL98N0

5. Simple GAN Implementation:  
https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4

#### Practice Material

Find in [W6 Folder](W6)