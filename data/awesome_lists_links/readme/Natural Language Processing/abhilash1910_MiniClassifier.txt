# Mini-Classifier

[![Read the Docs](https://readthedocs.org/projects/yt2mp3/badge/?version=latest)](https://yt2mp3.readthedocs.io/en/latest/?badge=latest)

## A Deep Learning Framework For Text Classification in Keras :robot:

This library is for Text/Intent/Semantic Classification comprising of 3 architectures-Bilstm,Lstm-Dense,Convolution Networks. The library can be used with pre-trained embeddings such as Glove,Fasttext,Word2vec and the file containing these have to be passed in as arguements. Without pretrained embeddings, the default Embedding layer from keras is used. For space separated textual labels, the labels should be modified by removing the spaces,and label encoding them before passing it into the models. For all other cases,the labels are automatically label encoded. The entire workflow is provided for 2 classification tasks- Binary and Categorical. These can be found in the 'BinaryClassificationTest.py' and 'CategoricalClassificationTest.py' files respectively. Most of the workflow is the same, with the only changes are in the dataset (the path of the dataset to be analysed should be used.), the hyperparameters for the models and the embeddings(if pre-trained embeddings are used, then the relative path to the pre-trained embedding file should be passed as arguement in the <modelname>.parameters() method). The Test scripts contain elaborate descriptions of using these.In the parameters() method in the models, the last 2 arguements specify using pre-trained embeddings or not. Specifying the second last arguement as False, implies that the model will not used any pretrained embeddings.Specifying it as True will allow the model to pick up the embedding file provided in the next(last) arguement for analysis/training.
This falls under sequence to sequence learning paradigm and this paper by [Illya etal](https://paperswithcode.com/paper/sequence-to-sequence-learning-with-neural) is the inspiration for this library.As of now, this architecture supports Glove Embeddings.

## Dependencies

<a href="https://www.tensorflow.org/">Tensorflow</a>


<a href="https://keras.io/">Keras</a>


## Usability

The library or the Layer is compatible with Tensorflow and Keras. Installation is carried out using the pip command as follows:

```python
pip install MiniClassifier==0.1
```

For using inside the Jupyter Notebook or Python IDE (along with Keras layers):

```python
import MiniClassifier.MiniClassifier as mc
```


## Running in Kaggle

This Kaggle [notebook](https://www.kaggle.com/abhilash1910/miniclassifier-library) provides an overview of using the library.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
