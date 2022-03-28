# Introduction
This repository has scripts and Jupyter-notebooks to perform all the different steps involved in [**Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer**](https://www.aclweb.org/anthology/D19-1322/)  
This mechanism is used for **_text style transfer_** when a **_parallel corpus_** for both the styles is not available. This mechanism works on the assumption that the text of any style is made of **two** parts: **1. Content** and **2. Attributes** . Below is a simpe example of a resturent review.
```
The food was great and the service was excellent.
Content: The food was and the service was
Attributes: great, excellent
```
We transfer the style of a given content in in two different ways. The first is referred to as the **Delete and Generate** approach (referred to as Blind Generative Style Transformer - B-GST in the paper) in which the model transfers the style of the text by choosing attributes automatically learnt during training. The second is referred to as the **Delete, Retrieve and Generate** (referred to as Guided Generative Style Transformer - G-GST in the paper) approach in which the model uses attributes retrieved from the target corpus to generate target sentences from the content. Below are a few examples.

**Generative a negative sentiment sentence from content (neutral) with Delete and Generate**
```
Content: The food was and the service was
Output: The food tasteless and the service was horrible.
```

**Generative a negative sentiment sentence from content (neutral) with Delete, Retrieve and Generate**
```
Content: The food was and the service was
Attributes: blend, slow
Output: The food was blend and the service was slow.
```
The names **Delete and Generate** and **Delete, Retrieve and Generate** are based on the steps involved in preparing training and test(reference) data. 
In **Delete and Generate**,  we prepare the training data by removing the attribute words from the text and during training teach the model to generate the sentence given content and target style. This is trained the same way a language model is trained. Below is an example.
```
The food was great and the service was excellent.
Content: The food was and the service was
Training input: <POS> <CON_START> The food was and the service was <START> The food was great and the service was excellent . <END>

The food was awful and the service was slow.
Content: The food was and the service was
Training input: <NEG> <CON_START> The food was and the service was <START> The food was awful and the service was slow . <END>
```
Cross entropy loss is calculated for all the tokens predicted after **_\<START\>_** token. For inference, we represent target style using the same tags as used in training, and provide the content as inputs to the model. For the case of sentiment style transfer, all the positive sentiment test data sentences will have **_\<NEG\>_** and all negative sentiment sentences will have **_\<POS\>_** token before the content. Below is an example.
```
Negative test data: <POS> <CON_START> the food was and the service was <START> 
Positive test data: <NEG> <CON_START> the food was and the service was <START> 
```

In **Delete, Retrieve and Generate**, we prepare the training data similar to the **Delete and Generate** but insted of target text style we specify the exact attributes to be used for generating the sentence from the content. Below is an example.
```
The food was great and the service was excellent.
Content: The food was and the service was
Training input: <ATTR_WORDS> great excellent <CON_START> The food was and the service was <START> The food was great and the service was excellent . <END>

The food was awful and the service was slow.
Content: The food was and the service was
Training input: <ATTR_WORDS> awful slow <CON_START> The food was and the service was <START> The food was awful and the service was slow . <END>
```
Otherwise the training is same as the **Delete and Generate**. During inference, to perform style transfer we need to get the attributes of opposite text style, we get it by retrieving similar content from opposite train corpus and use the attribute associated with that. Below can be a good example.   

```
Negative test data: <ATTR_WORDS> great tasty  <CON_START> the food was and the service was <START> 
Positive test data: <ATTR_WORDS> blend disappointing <CON_START> the food was and the service was <START> 
```


**The process of style transfer consists of multiple steps.** 

**_1. Prepare Training data_**
  * Train a classifier which uses attention mechanism. Here we have used [BERT](https://arxiv.org/abs/1810.04805) classifier.
  * Use attention scores to prepare data for **Delete and Generate** trainig and test.
  * Use the training and testing data of **Delete and Generate** to prepare training and test data for **Delete, Retrieve and Generate** .  
  
**_2. Generator Training_**
  * We have use modified version of [OpenAI GPT](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_openai_gpt.py) 
  * Run training of **Delete and Generate** and **Delete, Retrieve and Generate** . 
 
**_3. Generate sentences_**
  * Generate sentences from the test(reference) files.

The following section describes steps required from preparing the data to running inference. 

# Python Packages
1. torch     >=                1.0.1.post2
2. pandas  >=     0.23.4
3. numpy >=      1.15.4
4. python >=      3.7.1
5. tqdm    >=       4.28.1 
6. boto3 >= 1.9.107
7. requests >= 2.21.0
8. regex >= 2019.2.21

**Important Note**: We have used the BERT tokenizer which isn't the default. The default tokenizer is using spacy library. To replicate the exact results please run in the environment where spacy isn't installed. Having spacy installed will tokenize even the special tokens, and lead to different tokens than we have used, leading to unpredictable results.

# Steps
### 1. Classifier Training:

We have used [BERT](https://arxiv.org/abs/1810.04805) for classification. This classification trainings helps to find the attributes from the sentence. We choose one particular head of BERT model, for which the tokens which have high attention weights are those that are stylistic attributes of the sentence. 
  * Classifier Training Data Preparation: **_BERT_Classification_Training_data_preparation.ipynb_** notebook creates training, testing and dev data. Modify the the paths of the input and output files.
  * BERT Classifier Training: Run the below command to train BERT classifier.
  ```bash
  export BERT_DATA_DIR=Path of the directory containing output previous step (train.csv, dev.csv)
  export BERT_MODEL_DIR=Path to save the classifier trained model
  ```
  
  ``` bash
  python run_classifier.py \
  --data_dir=$BERT_DATA_DIR \
  --bert_model=bert-base-uncased \
  --task_name=yelp \
  --output_dir=$BERT_MODEL_DIR \
  --max_seq_length=70 \
  --do_train \
  --do_lower_case \
  --train_batch_size=32 \
  --num_train_epochs=1 
  ```
We ran this with a single K80 Tesla GPU with 12 GB GPU Memory (AWS p2.xlarge instance). The batch size can be modified based on the max_seq_length. The code can be used with multiple GPUs and batch size can be increased proportanally. For p2.8xlarge, suggested train_batch_size = 256 and for p2.16xlarge, train_batch_size=512.


### 2. Selecting Attention Head:
  * Run **_Head_selection.ipynb_** for finding out which attention heads capture attribute words properly. The final result will be a list of tuples (Block/layer, Head, Score) sorted from best to worst. Use the first Block/layer and Head combination in the further steps.

### 3. Prepare training and inference data:
  * Run **_BERT_DATA_PREPARATION.ipynb_** for preparing training and inference data for **Delete and Generate** . Use the best layer, Head combination from the previous step in **run_attn_examples()** function.
  * Run **Delete_Retrieve_Generate_Data_Preparation.ipynb** to generate training data for **Delete, Retrieve and Generate** . It generates train, dev and test files. Use the files generated by **process_file_v1()** function as it shuffles the attributes and randomly samples only 70% of the attributes to train the generator model to generate smooth sentences instead of teaching it to just fill the blanks.
  * Run **tfidf_retrieve.ipynb** to generate inference data by retrieving attributes of closest match from target style training corpus. 
  
### 4. Generator Model training:
  * Run **_openai_gpt_delete_and_generate.py_** for training **Delete and Generate** model. Below is the sample command.
  ```bash
  export DG_TRAIN_DATA=Path to the training file generated in the previous step
  export DG_EVAL_DATA=Path to the eval file generated in the previous step
  export DG_MODEL_OUT=Path to save the Delete and Generate model weights
  ```
  ```bash
  python openai_gpt_delete_and_generate.py \
  --model_name openai-gpt \
  --do_train \
  --do_eval \
  --train_dataset $DG_TRAIN_DATA \
  --eval_dataset $DG_EVAL_DATA \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 85 \
  --output_dir $DG_MODEL_OUT 
  ```

  * Run **__openai_gpt_delete_retrive_and_generate.py_** for training **Delete, Retrieve and Generate** model. Below is the sample command.
```bash
  export DRG_TRAIN_DATA=Path to the training file generated in the previous step
  export DRG_EVAL_DATA=Path to the eval file generated in the previous step
  export DRG_MODEL_OUT=Path to save the Delete, Retrieve and Generate model weights
  ```
  ```bash
  python openai_gpt_delete_retrive_and_generate.py \
  --model_name openai-gpt \
  --do_train \
  --do_eval \
  --train_dataset $DRG_TRAIN_DATA \
  --eval_dataset $DRG_EVAL_DATA \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 85 \
  --output_dir $DRG_MODEL_OUT 
  ```

This configuration is with 1 K80 Tesla GPU with 12 GB GPU Memory (AWS p2.xlarge instance). The batch size can be modified based on the max_seq_length. The code can be used with multiple GPUs and batch size can be increased proportionally. For p2.8xlarge, suggested train_batch_size = 256 and for p2.16xlarge, train_batch_size=512. **All the sentences with number of tokens > max_seq_length will be removed from the training.**

### 5. Style transfer on test data:
  * Run **_OpenAI_GPT_Pred.ipynb_** for generating style transfer on the test data.
  
# Results (outputs):
All the outputs on the test (reference) sets for each of the 5 datasets are present [here](https://github.com/agaralabs/transformer-drg-style-transfer/tree/master/results). Each directory is named after a dataset and contains 2 csv files - one for each direction of style transfer. In each of the csv files, the column named **BERT_DEL** has outputs of our B-GST model and the column named **BERT_RET_TFIDF** has outputs of our G-GST model. **These results can be used for replicability.**
  
# Models
As an example, we have shared the trained model weights for Yelp dataset for easy testing on test dataset. It can be downloaded from [here](https://drive.google.com/open?id=1tTvwm_MafHxTMhwo93VvzK_EeGoUpFx8) and directly use in **step 5**.

# Datasets
### 1. Yelp, Amazon & ImageCaption:
The dataset we have used can be downloaded from https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data

### 2. Political:
The dataset we have used can be downloaded from http://tts.speech.cs.cmu.edu/style_models/political_data.tar

### 3. Gender:
The dataset we have used can be downloaded from http://tts.speech.cs.cmu.edu/style_models/gender_data.tar

# Citation
If you use our reseach or code please cite the work,
```
@inproceedings{Sudhakar2019TransformingDR,
  title={Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer},
  author={Akhilesh Sudhakar and Bhargav Upadhyay and Arjun Maheswaran},
  booktitle={EMNLP/IJCNLP},
  year={2019}
}
```
