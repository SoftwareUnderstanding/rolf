# SelfDiscovery
SelfDiscovery is a BERT model based Question Answer System ,where given a context system can give you the sollution to asked queries.  


PyTorch pretrained bert model.

pip install pytorch-pretrained-bert

Paper:https://arxiv.org/abs/1810.04805

Citation. @article{devlin2018bert, title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}, author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina}, journal={arXiv preprint arXiv:1810.04805}, year={2018} }

Huggingface implimentation is taken as refference for the pytorch implimentation. https://github.com/huggingface/pytorch-pretrained-BERT

Ussage.
1) set output_dir variable to correspoding your local directory
2) set train_file to train file path(i.e train-v1.1.json) in json format. Check the folder data for the sample formats
3) set predict_file to predict file format.
4) If you have GPU installed set no_cuda=False.
5) set predict_batch_size and train_batch_size as per you need

Inorder to use pretrained model
1) set MODEL_SAVED_LOAD to true. 
2) set do_train is false and do_predict is true.
3) Place the pretrained model in output_dir

Right Now I have trained the model in bin folder.
Thanks to SQUAD https://rajpurkar.github.io/SQuAD-explorer/ for the datasets.

In ordered to train your own data.
1) For example train-v1.1.json is kept under datasets folder. Prepare the dataset of the same format and
2) set MODEL_SAVED_LOAD to false. 
3) set do_train is True and do_predict is False

Once the model is trained, trained .bin file will be stored in output_dir.

Prediction.
1) set MODEL_SAVED_LOAD to true. 
2) set do_train is false and do_predict is true.

Check the prediction accuracy and predictions.json is under outdirectory.







