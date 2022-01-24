# OP Text
A wrapper around the popular [transformers](https://github.com/huggingface/transformers)  machine learning library, by the HuggingFace team. OP Text provides a simplified, [Keras](https://keras.io/) like, interface for fine-tuning, evaluating and inference of popular pretrained BERT models.

## Installation
PyTorch is required as a prerequisite before installing OP Text. Head on over to the [getting started  page](https://pytorch.org/get-started/locally/) of their website and follow the installation instructions for your version of Python. 

>!Currently only Python versions 3.6 and above are supported

Use one of the following commands to install OP Text:

### with pip

    pip install op_text
    
### with anaconda

    conda install op_text
    
## Usage 
The entire purpose of this package is to allow users to leverage the power of the transformer models available in HuggingFace's library without needing to understand how to use PyTorch.

### Model Loading

Currently the available models are:

 1. **BERT** from the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(https://arxiv.org/abs/1810.04805)](https://arxiv.org/abs/1810.04805) released by Google.

2. **RoBERTa** from the paper [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) released by Facebook.

3. **DistilBERT** from the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) released by HuggingFace.

Each model has contains a list of the available pretrained models.
Use the **DOWNLOADABLES** property to access them.

    Bert.DOWNLOADBLES
    >> ['bert-base-uncased','bert-large-uncased']
    
	Roberta.DOWNLOADBLES
	>> ['roberta-base','roberta-large']
	
	DistilBert.DOWNLOADABLES
	>> ['distilbert-base-uncased']

Loading a model is achieved in one line of code. The string can either be the name of a pretrained model or a path to a local fine-tuned model on disk. 

    from op_text.models import Bert
	
	# Loading a pretrained model 
	model = Bert("bert-base-uncased", num_labels=2)
	
	# Loading a fine-tuned model
	model = Bert("path/to/local/model/")

> 	Supply *num_labels* when using a pretrained model, as an untrained classification head is added when using this one of the DOWNLOADABLE strings.

###  Fine-tuning
Finetuning a model is as simple as loading a dataset, instantiating a model and then passing it to the models *fit* function. 

    from models.import Bert
	    
	X_train = [
		"Example sentence 1"
		"Example sentence 2"
		"Today was a horrible day"
	]
	y_train = [1,1,0]
	
	model = Bert('bert-base-uncased', num_labels=2)
	model.fit(X_train, y_train)

### Saving
At the conclusion of training you will most likely want to save your model to disk. Simply call the the models *save* function and supply an output directory and name for the model.

    from models.import Bert
	model = Bert('bert-base-uncased', num_labels=2)
	model.save("path/to/output/dir/", "example_save_name")

### Evaluation 
Model evaluation is basically the same as model training. Load a dataset, instantiate a model and but instead call the *evaluate* function. This returns a number between 0 and 1 which is the percentage of predictions the model got correct.

    model.evalaute(X_test, y_test)
    >> 0.8 # 80% correct predictions

### Prediction

Predict the label of a piece of text/s by passing a list of strings to the models 
*predict* function.  This returns a list of tuples, one for each piece of text to be predicted. These tuples contain the models confidence scores for each class and the numerical label of the predicted class. If a label converter is supplied, a string label of the predicted class is also included in each output tuple.
	
    from op_text.utils import LabelConverter
	
	converter = LabelConverter({0: "negative", 1: "positive"}    
    to_predict = ["Today was a great day!"]
	model.predict(to_predict, label_converter=converter)
	>> [([0.02, 0.98], 1, "positive")]

## Citation

Paper you can cite for the  Transformers library:
```
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```