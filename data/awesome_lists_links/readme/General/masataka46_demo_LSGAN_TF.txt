# demo program of LSGAN using tensorflow  
This is the demo program of LSGAN using tensorflow.  
It needs input data like Food101 datasets.  

# literature  
X. Mao, et. al. "Least Squares Generative Adversarial Networks"  
https://arxiv.org/pdf/1611.04076.pdf  

# dependency  
I confirmed operation only with..  
1)python==3.6.0    
2)tensorflow==1.8.0   
3)matplotlib==2.0.0  

# architecture  
The architecture of LSGAN is based on https://github.com/hei4/DLRepertoire03.  

# computation graph  
1) over all  
![tensorboard_lsgan_01_all](https://user-images.githubusercontent.com/15444879/42793425-0be7640c-89b5-11e8-8b24-3b0f7284aae6.png)  

2) around generator  
![tensorboard_lsgan_02_g](https://user-images.githubusercontent.com/15444879/42793432-11e667f4-89b5-11e8-908b-884d65663bde.png)  

3) around discriminator  
![tensorboard_lsgan_03_d](https://user-images.githubusercontent.com/15444879/42793440-18c2a3c6-89b5-11e8-998e-5a2dc506af71.png)  

4) around loss  
![tensorboard_lsgan_04_loss](https://user-images.githubusercontent.com/15444879/42793451-1f2d0080-89b5-11e8-932b-466d68f049dd.png)  

# Prediction  
generated images are below.  
ex1)  Food101/dumplings  
![resultimage_dumplings01_990](https://user-images.githubusercontent.com/15444879/42793211-0344f4c8-89b4-11e8-9062-37f7edfffa09.png)  

ex2)  Food101/paella  
![resultimage_paella01_990](https://user-images.githubusercontent.com/15444879/42793229-1c831a0a-89b4-11e8-8ee9-d21b0e90dd86.png)  

ex3)  Food101/pho  
![resultimage_pho01_990](https://user-images.githubusercontent.com/15444879/42793237-237a8d34-89b4-11e8-95b8-580481c7f9e7.png)  

ex4)  Food101/sushi  
![resultimage_sushi01_990](https://user-images.githubusercontent.com/15444879/42793242-2884a08a-89b4-11e8-8954-6dfda042dd28.png)  