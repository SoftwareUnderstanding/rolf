# sentiment-Analysis-for-German-Datasets
Google


PyTorch pretrained bert model.

pip install pytorch-pretrained-bert



Paper:https://arxiv.org/abs/1810.04805

Citation.
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

Huggingface implimentation is taken as refference for the pytorch implimentation.
https://github.com/huggingface/pytorch-pretrained-BERT


Ussage:: 
output_dir set as per your directory.
depends on need of  accuracy change the value of num_train_epochs.
task_name
Download the data from the below websites.

Datasets:: https://www.nyu.edu/projects/bowman/xnli/
           https://sites.google.com/view/germeval2017-absa/data
           https://sites.google.com/site/iggsahome/downloads
  
           
Keep the data in data_dir. Right now script supports .tsv and .xml file formatt.
tsv file formatt as follows
SportBrÃ¼che am Becken und an den Rippen so die Nachricht des Crashpiloten.	Sport
sentence(SportBrÃ¼che am Becken und an den Rippen so die Nachricht des Crashpiloten) and category(Sport) should be separated by a tab.


Format of xml file is given in the data folder. Check the sampleXml.xml file for the format.

Training:: 
set MODEL_SAVED_LOAD as False 
set do_train as True
set do_eval as False
Once the model is trained fully,it will be stred under output folder as bin file format.

Evaluating::
set MODEL_SAVED_LOAD as True 
set do_train as False
set do_eval as True

previously saved model will be loaded find evaluation can be done.

For example if you want to classify into 10 category go to
num_labels_task dictionary in code and set the corresponding item value to 10.
class BBC5Processor support xml format all other support tsv format.


           
         
           
         


