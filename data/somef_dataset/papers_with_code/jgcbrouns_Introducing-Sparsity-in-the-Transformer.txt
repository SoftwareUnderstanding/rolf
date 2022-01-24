
# Introducing Sparsity in the Transformer model (Keras Implementation)

A proof of concept implementation of evolutionary sparsity in the Transformer model architecture.  

## How To Run:
### Sparse Variant of Transformer
*Sparse variant architecture, trained on the original data (29.000 samples in training set, 1024 samples in test set)*
```
python3 en2de_main.py sparse origdata
```
### Original Transformer
*Original architecture with a rewritten trainingsloop and using custom transfer-function in order to validate the obtained results *
```
python3 en2de_main.py originalWithTransfer origdata
```
### Flags:
#### load_existing_model
*Loads the saved model from previous training epochs and continues training this model*

#### Datasets
*sets the dataset to be used for the trainings-task*
- **'origdata':** *Use the WMT 2016 German-to-English dataset for training*
- **'testdata':** *Use a very small subset of the original trainings-task*



## Research papers:
- **The Transformer original paper:**  
"[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017)
- **SET-procedure original paper:**  
[Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science](https://www.nature.com/articles/s41467-018-04316-3) (Decebal Constantin Mocanu, Elena Mocanu, Peter Stone, Phuong H. Nguyen, Madeleine Gibescu & Antonio Liotta)

## Code based on / uses parts of:
- **Transformer implementation in Keras by LSdefine:**  
[The Transformer model in Attention is all you need：a Keras implementation.](https://github.com/Lsdefine/attention-is-all-you-need-keras)
- **Attention is all you need - A Pytorch implementation**  
[Jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
- **Sparsity SET-procedure based on the proof-of-concept code of:**   
[Dr. D.C. Mocanu - TU/e](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/blob/master/SET-MLP-Keras-Weights-Mask/fixprob_mlp_keras_cifar10.py)

## F.A.Q
- The test sys argument gives me error: UnicodeEncodeError: 'ascii' codec can't encode character '\xe4' in position 6: ordinal not in range(128).   
**Solution:** *run in terminal:* ```export LC_CTYPE=C.UTF-8```
  
  

