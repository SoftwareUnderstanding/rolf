# Variational Auto-Encoder (VAE) 

PyTorch re-implementation of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma et al. 2013.

## download dataset
MINST http://yann.lecun.com/exdb/mnist/

FreyFace https://cs.nyu.edu/~roweis/data.html

save to './datasets/'

or specify the arguments



## quick start
```shell
python ./src/VAE_train.py \
--data='FreyFace' \
--data_path='./datasets/FreyFace' \
--batch_size=100 \
--latent_dim=10 \
--hidden_dim=200 \
--learning_rate=0.01 \
--epoch=10000 \
--output_dir='./output'
```

## help
```shell
python ./src/VAE_train.py -h
```

## citation
```
@ARTICLE{2013arXiv1312.6114K,
       author = {{Kingma}, Diederik P and {Welling}, Max},
        title = "{Auto-Encoding Variational Bayes}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Machine Learning, Computer Science - Machine Learning},
         year = "2013",
        month = "Dec",
          eid = {arXiv:1312.6114},
        pages = {arXiv:1312.6114},
archivePrefix = {arXiv},
       eprint = {1312.6114},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013arXiv1312.6114K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
