# Unsupervised Representation Learning
Explores different types of autoencoders as pre-text training models for transfer 
learning on computer vision tasks.

## files to look at
1. ae.py : Includes concepts/inspiration from multiple papers  
    1. https://arxiv.org/pdf/2002.05709.pdf
    2. https://arxiv.org/pdf/1807.07543.pdf
    3. https://arxiv.org/pdf/1606.05908.pdf
    4. https://arxiv.org/pdf/1706.00409.pdf
2. modules.py : definition of all models
3. train_utils.py : training utility functions for loading image data, logging 
    intermediate results and loss values and computing loss values.
4. ae_job_dispatcher.py : SLURM job dispatching program to help perform 
    distributed grid search autoencoder hyperparameters.

Code cleanup remaining!