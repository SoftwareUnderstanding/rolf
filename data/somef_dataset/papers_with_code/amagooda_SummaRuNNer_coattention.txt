#Implementation of the paper "Attend to the beginning: A study on using bidirectional attention for extractivesummarization" To appear in (FLAIRS33).
https://arxiv.org/pdf/2002.03405.pdf


It is a modification of the original SummaRuNNer Paper (https://arxiv.org/pdf/1611.04230.pdf) for discussion Thread summarization.

##1- Modifications
###a- Post to comment Co-attention:
Co-attention mechanism is added between initial post sentence representations and each comment sentence representations. 
The co-attention produces a set of post aware comment representations.
###b- Pretraining:
Model can be pretrained using different dataset, and then fine-tuned with different dataset.
 The finetuning can be done with the pretrained vocab or the embedding layer can be reinitialized with different vocab.
###c- Gradual layer unfreezing:
Gradual layer unfreezing is integrating in this code influenced by https://arxiv.org/pdf/1801.06146.pdf.

Layers are freezed at the begining of the training process, then layers are gradually unfreezed starting from the top most layer down to the embedding layer. 
###d- Backtranslation (under development):

--------------------------------
##2- How to use:

###a- Data preprocessing:
To either train, tune or test a model first you need to preprocess your data. You can run the preprocess.py file to achieve that.
To run the _preprocess.py_, you can implement your data reading function in _data_loader.py_.

The reading function should read the data split into train, val, test parts and convert to the thread object. please refer to _data_loader.py_ for examples.

In _preprocess.py_ adjust these parameters to your liking :

`params['DATA_Path'] = './cnn_data/finished_files/' ` The path to your data

`params['data_set_name'] = 'cnn''` dataset name, appended to saved checkpoint and output.

`params['use_BERT'] = True/False` use Bert embedding or not, using Bert embedding takes longer for preprocessing.

`params['BERT_Model_Path'] = '../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/'` The path to the bert model

`params['BERT_embedding_size'] = 768` The size of bert embeddings

`params['BERT_layers'] = [-1] or [-1, -2] or [-1, -2, -3], etc..` The indcies of bert layers to be used, where a word representation is the concatination of these layers.

`params['vocab_size'] = 70000` The maximum size of vocab to use.

`params['use_back_translation'] = False` use back-translation or not

`params['back_translation_file'] = None` back-translated file path

`params['Global_max_sequence_length'] = 25` maximum number of tokens to keep in a sequence,
 longer sequences will be truncated and shorter ones will be padded

`params['Global_max_num_sentences'] = 20` maximum number of sentences in a comment, longer comments will be truncated.

`params['use_external_vocab'] = False` use external vocab to encode the data ? if False the data will be 
encodeded using the vocab extracted during the preprocessing.

`params['external_vocab_file'] = './checkpoint/forum_vocab.pickle'` external vocab file path.

`params['encoding_batch_size'] = 64` The batch size used in encoding data, only used when using Bert for encoding.

`params['data_split_size'] = 15000` The size of data chunk to be processed at a time, if data is larger than _params['data_split_size']_ samples,
then preprocessed data will be split into multiple pieces.

`params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` use cpu or cuda


###b- Train a model
To train a model using the preprocessed data you can run the _main.py_. Make sure to 
adjust the parameters to your preferences.

`params['DATA_Path'] = `

`params['data_set_name'] = 'cnn'`

`params['use_coattention'] = False` use co-attention or not.

`params['use_BERT'] = False` use BERT embeddings or not.

`params['BERT_embedding_size'] = 768`

`params['BERT_layers'] = [-1]`

`params['embedding_size'] = 64` Embedding size, if BERT not used.

`params['hidden_size'] = 128` Hidden size for sentence and document RNNs

`params['batch_size'] = 8` The batch size for training.

`params['lr'] = 0.001` Learning rate.

`params['vocab_size'] = 70000` max vocab size.

`params['Global_max_sequence_length'] = 25, params['Global_max_num_sentences'] = 20` Use the same as the ones used in preprocessing

`params['num_epochs'] = 50` Num of training epoch

`params['start_epoch'] = 0` The starting epoch index, should be helpful to avoid overwriting saved checkpoints.

`params['write_summarizes'] = True` 

`params['output_dir'] = './output/'`

`params['save_model'] = True`

`params['save_model_path'] = './checkpoint/models/'`

`params['load_model'] = True`

`params['reinit_embeddings'] = True`

`params['load_model_path'] = './checkpoint/bilstm_model_cnn_19.pkl'`

`params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

`params['task'] = 'Train'  ### Train, Test`

`params['write_post_fix'] = '0'`

`params['tune_postfix'] = '_tune_guf'`

`params['gradual_unfreezing'] = True`

###c- use a pretrained model for Tuning
To tune, run main.py after setting the following parameters.

`params['load_model'] = True` Set for True to load a checkpoint.

`params['reinit_embeddings'] = True` True if the embedding layer needs to be reinitialized, 
or False to use the same embedding layer used while training.

`params['load_model_path'] = './checkpoint/bilstm_model_cnn_19.pkl'` The checkpoint to load for tuning

`params['task'] = 'Train'  ### Train, Test` set True to continue training

`params['tune_postfix'] = '_tune_guf'` a postfix to add for checkpoint saving and output files.

`params['gradual_unfreezing'] = True` use gradual unfreezing while tuning.

###d- Use a checkpoint for Testing
