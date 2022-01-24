# Deep comedy
## Report
### RNN char level: `Embedding - LSTM - Dense - Dense`
- Good metric results
- Prone to overfitting
- No rhymes
- No padding
### RNN syllabe level: `Embedding - LSTM - Dense - Dense`
- https://www.groundai.com/project/neural-poetry-learning-to-generate-poems-using-syllables/1
- Very bad results
- not converging loss
- padding on/off
### RNN syllabe level: `LSTM - Dense - Dense`
- Removing Embedding gives a slightly improvement, but not yet comparable to the char level method
- Uneffective metric loss
### 3 Input Char model with 1 `LSTM`:
- https://towardsdatascience.com/generating-haiku-with-deep-learning-dbf5d18b4246
- slightly improvement, but not yet comparable to the char level method
- Good performances with syllable count
- No Rhymes
### 3 Input Syllable model with 1 `LSTM`:
- https://towardsdatascience.com/generating-haiku-with-deep-learning-dbf5d18b4246
- Significantly worse than the 3 Input Char model
- No Rhymes
- Bad syllable counting
### Transformer
- https://arxiv.org/pdf/1706.03762.pdf
- (generation) https://huggingface.co/blog/how-to-generate
- explanation pos encoding https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
