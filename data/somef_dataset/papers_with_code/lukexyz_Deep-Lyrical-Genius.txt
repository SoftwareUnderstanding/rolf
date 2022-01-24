# Lyrical-Genius
Transformer NLP models on Top 40 lyrics database.

DistilBERT (2019), a distilled version of BERT: smaller, faster, cheaper and lighter  
[https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)


### Install Notes
Python 3.6 required for pytorch and spacy-transformers  
```> conda create -n deeplyrics python=3.6 pip jupyter nb_conda_kernels ipywidgets```  
```> conda activate deeplyrics```  
```> conda update jupyter_core jupyter_client```  

##### `Pytorch`  
* Get CUDA version from `nvcc --version` on windows, or `$ nvidia-smi` on unix.  

For conda on windows and Cuda 9, use pytorch instructions on [https://pytorch.org/](https://pytorch.org/), pip install probably works fine on unix.  

```> conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev```



##### `spacy-transformers`
For GPU installation, find your CUDA version add the version in brackets, e.g. spacy-transformers[cuda92] for CUDA9.2 or spacy-transformers[cuda100] for CUDA10.0.   
[spacy-transformers](https://github.com/explosion/spacy-transformers#-quickstart)

```> pip install spacy-transformers[cuda92]```





#### Create environment file
```conda env create -f environment.yml```
