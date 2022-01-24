# PSSP
Protein Secondary Structure Prediction project

Final Project for the course of Machine Learning 2020-2021, University of Florence. 
The project aims to deepen the techniques and concepts exposed during the course through the implementation of the architectures and some tests. 
The task we will try to solve is the prediction of the secondary structure of the protein in 8 classes mainly with an architecture based on RNNs and, in the end, an attempt to adapt the task to the Transformer architecture. 

For more details, a report in Italian is available [here](https://github.com/emanuele-progr/PSSP/blob/main/report_ita/MLProject.pdf). 

![struttura-delle-proteine (1)](https://user-images.githubusercontent.com/22282000/123549058-abde2d80-d767-11eb-98fb-9eff75d15c6c.jpg)

The reference dataset refers to Zhou & Troyanskaya's work and can be downloaded from the page https://www.princeton.edu/~jzthree/datasets/ICML2014/
*********************************************************************************************************************************

## SOURCES 

With changes and additions, this code starts from : 

- For the prepared dataset and baseline concepts : TY - CPAPER TI - Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction AU - Jian Zhou AU - Olga Troyanskaya BT - Proceedings of the 31st International Conference on Machine Learning PY - 2014/01/27 DA - 2014/01/27 ED - Eric P. Xing ED - Tony Jebara ID - pmlr-v32-zhou14 PB - PMLR SP - 745 DP - PMLR EP - 753 L1 - http://proceedings.mlr.press/v32/zhou14.pdf UR - http://proceedings.mlr.press/v32/zhou14.html 

- For the base architecture : "Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks" / 2560 
   Zhen Li, Yizhou Yu https://www.ijcai.org/Abstract/16/364
  
- For the concepts and Transformer implementation : "Attention Is All You Need" 
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin 
  https://arxiv.org/abs/1706.03762
  
  "The Annotated Transformer" Alexander Rush 
  http://nlp.seas.harvard.edu/2018/04/03/attention.html 
  
  Transformer implementation and preprocessing by
  https://github.com/OpenNMT/OpenNMT-py
  
  
  
 *********************************************************************************************************************************



