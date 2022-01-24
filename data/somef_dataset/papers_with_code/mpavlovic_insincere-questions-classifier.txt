# Insincere Questions Classifier
Deep learning model for the Quora Insincere Questions Classification competition on Kaggle. Based on bidirectional LSTMs, GRUs and [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf). Implemented in TensorFlow 2.0 and Keras.

| Public F1 Score  | Private F1 Score |
| :--------------: |:----------------:| 
| 0.69536          | 0.70140          |

## Model
The model is in essence a logistic regression classifier whose inputs are featrures extracted in previous layers: 
<br>
<img src="model.PNG" width="640">
<br>
The input to the model are two 300 dimensional weighted sums of pretrained embeddings available on the competition, concatenated to a single 600 dimensional input vector passed to the embedding layer. Embeddings were frozen during training.<br><br>
After spatial dropout, a bidirectional LSTM layer is applied, whose states are averaged and max pooled. Besides that, the last output state is passed to the Bahdanau attention layer as a query, together with all states over time as values. This was done separately for the first 256 dimensions of the output states, which are result of the left-to-right LSTM pass. The same thing was repeated for the second 256 dimensions of the LSTM states (right-to-left pass).<br><br>
The second middle layer is a bidirectional GRU, but implemented as two separate layers - one in the LTR, and other in RTL direction. This may look as a weird choice, but couple of experiments constantly showed better performance when implemented this way. The GRU states from both directions were averaged and max pooled, as well as passed to the respective Bahdanau attention layers as values, together with last output states as queries.<br><br>
All average, max pool and attention outputs are concatenated and passed to a single neuron in the output dense layer. Vectors from LSTM layer are passed through a skip connection over the GRU layers. Output layer is effectively a logistic regression classifier whose input is a vector of extracted features from different network layers. 

## Applied Techniques
**Preprocessing for embeddings coverage** - it was very important to preprocess text to ensure as higher embeddings coverage as possible. These techniques included cleaning of contractions and special characters, simpler spelling correction, lowercasing of words whose uppercased versions didn't exist in embeddings and/or usage of stemmers and lemmatizer. See References for more details about preprocessing.<br><br>
**Weighted averaging and concatenation of embeddings** - the model was firstly trained with 0.7 * GloVe + 0.3 * Paragram embeddings, as it was shown [here](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568). Other two embeddings were averaged later (0.7 * WikiNews + 0.3 * GoogleNews) and concatenated to the first ebmeddings average. Both techniques improved model performance.<br><br>
**Custom threshold search** - finding of the right decision threshold was also one of the crucial steps in obtaining high leaderboard result. The search was done with the help of 5-folded cross validation - after training on each fold, a threshold with highest F1 score was retained. Thresholds from all folds were later averaged to the single final threshold value.<br><br>
**Attention** - the attention mechanism was added after each recurrent layer. Implemented version was from Bahdanau et al. (2015). Original implementation is available [here](https://www.tensorflow.org/tutorials/text/nmt_with_attention).<br><br>
**Model weights averaging** - this approach significantly improved the model performance - both cross validation loss and F1 score. First the right number of training epochs was chosen with the help of CV, and later the custom callback was implemented to average the model weights from last two epochs of training (third and fourth). Those averaged weights were set back to the model and final (submission) prediction was made.

## Usage & Requirements
Please install the following requirements:
* Python 3.6
* TensorFlow 2.0
* Numpy
* Pandas
* Scikit-learn
* Gensim
* NLTK
<br><br>

The submission script which was executed as a Kaggle kernel is `submission_script.py` file. It Is also available 
[here](https://www.kaggle.com/milanp/quora-insincere-questions-late-submission-script/code?scriptVersionId=23933427) as a public kernel.
<br><br>
If you want to make experiments with different models and hyperparameters, please use `main.py` and `build_model.py` files. The model can be tweaked in `build_model.py` file, whereas 5-folded cross validated experiment is executed in `main.py`. Before running, please update the appropriate paths to train and test datasets (lines 37 and 38), as well as paths to embedding files (lines 40-43) in `main.py`. Both training data and pretrained embeddings are available on the competition's [official website](https://www.kaggle.com/c/quora-insincere-questions-classification/). You can also change some hyperparameters in `hparams` dictionary in lines 97-120 of `main.py`. 
<br><br>
Settings from every experiment will be saved to a separate folder, which will be printed out at the end. Both `main.py` and `build_model.py` files are modified and belong to a small competition framework which is described in detail [here](https://github.com/mpavlovic/toxic-comments-classification).

## References
https://arxiv.org/pdf/1409.0473.pdf <br>
https://www.tensorflow.org/tutorials/text/nmt_with_attention <br>
https://www.kaggle.com/wowfattie/3rd-place <br>
https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings <br>
https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2 <br>
https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568 <br>

