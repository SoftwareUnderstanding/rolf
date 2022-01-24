# PGSG at SemEval-2020 Task 12
This repo contains the code for our solutions of SemEval-2020 Task 12 challenge, which won the **second place (2nd)** in *sub-task B: Automatic categorization of offense types* and were ranked 55th with a macro F1-score of 90.59 in *sub-task A: Offensive language identification*

## System Description
### Pretrained BERT with Tweets data
Due to the limitation of computational power, we decide to not pre-train BERT model from scratch but fine-tune from the BERT-Large, Uncased (Whole Word Masking) checkpoint.

In BERT’s vocabulary, there are 994 tokens marked as ‘unused’ which are effectively randomly initialized. We replace 150 of them with the top occurrences and offensive-related words of the Tweets dataset.

We use 9 milion tweet sentences to pre-train this BERT model. We follow the instruction of pre-training model from Google BERT github. However, since tweets data are single short sentences, we modify the processing and training script to remove the Next Sentence Prediction loss and only perform the Masked LM task.

Both Tensorflow and Pytorch checkpoint are released [here](https://bit.ly/3dpaTX7).

### BERT-LSTM model

In addition to the output vector of the [CLS] token from BERT model, in our implementation, the output vectors of all word tokens are also used for classification. Those tokens are sent through LSTM layers, then concatenated with the [CLS] token and finally passed to a fully connected neural network to perform the final classification
![Model Architecture](mics/model_structure.png)

### Noisy Student training
To leverage the enormous semi-supervised data given in the challenge, we use the [Noisy Student training method](https://arxiv.org/abs/1911.04252) to train the model.

We only select the most confidence instances from the training set and assign hard-label (NOT/OFF, TIN/UNT). These instances are used to train the ‘Teacher’ model.

Then we split the unlabeled data set to multiple subsets. At each iteration, we use the ‘Teacher’ model to score one subset to generate the pseudo labels and use the pseudo labels to train the ‘Stu- dent’ model. Finally, we iterate the process by putting back the student as a teacher to generate pseudo labels on a new subset and train a new stu- dent again.


## Results
| System | Macro-F1 |
|:------:|:------: |
|Sub-task A| 90.59 |
|Sub-task B| 73.62 |

## How to use

### Preprocess Data
```
python run_preprocess.py --data_file <raw data> --save_file <save file> --vocab_file <bert vocab>
```

### Train model
Update the `config.py` to choose the data to train, teacher model and training scheme
```
python run_training.py
```

### Generate label
Generate label to train student model, or generate final output
```
python run_inference.py --model_path <teacher model path> --inference_file <data to generate label (see `FILENAME` in config.py)> --output_file <output file>
```