# pytorch-facebook-udacity-challenge    


   PyTorch implementation of a Deep learning network to identify 102 different types of flowers (PyTorch Scholarship Challenge).   

   The used data set contains images of flowers from 102 different species divided in a training set and a validation set.The images can be downloaded [here](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip)   

  In addition, the repository contains a utility for testing the performance of a model on the [original flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) or alternatively on a dataset obtained by downloading the first 10 resulting images from Google, querying it by the name of the flower categories.   
  
  
  ## Overview    
  
  The repository contains 3 different implementation techniques :      
  
  1. **image-classifier-project-with-sgdr-warm-restarts**     
      This is based on the following [paper](https://arxiv.org/abs/1608.03983). The basic idea contains the following points :    
      a.  Reset the learning rate every so many iterations so that the model may be able to more easily pop out of a local minimum if it appears to be stuck.  
      b.  We know we will most likely get closer to a global minimum the more iterations we do through our dataset, we need a way to lengthen the time spent decreasing our learning rate. Instead of restarting every epoch, we can lengthen the number of epochs in a multiplicative way so that the first cycle will decrease over the span of 1 epoch, the second over the span of 2 epochs, and the third using 4 epochs, etc. 

      
  2. **resnet50-with-snapshots**      
    This is based on the following [paper](https://arxiv.org/abs/1704.00109)   
    The implementation contains ways to train once and get m models, giving an ensemble of models by training just once.


  3. **image-classifier-project-resnet50-final**     
     This is the final implememntation with the best validation accuracy for the model. Steps followed for this are :     
     a. Enable data augmentation, and precompute=True  
     b. find highest learning rate where loss is still clearly improving  
     c. Train last layer from precomputed activations for 1-2 epochs  
     d. Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1  
     e. Unfreeze all layers   
     f. Set earlier layers to 3x-10x lower learning rate than next higher layer   
     g. Train full network with cycle_mult=2 until over-fitting   
     
     

     
