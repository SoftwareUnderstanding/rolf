Tensorflow implementation of Teaching machine read and comprehend

introduce and tutorial[http://rsarxiv.github.io/2016/06/18/教机器学习阅读/]

This implementation contains:

Google DeepMind's [Teaching Machines to Read and Comprehend](http://arxiv.org/pdf/1506.03340v3.pdf):

	1. Deep LSTM Reader
	  
	2. Attentive Reader 
		- with [Bidirectional LSTMs](http://www.cs.toronto.edu/~graves/nn_2005.pdf) with peephole weights
		
Facebook:

	3. End 2 End Memory Network
    - End-To-End Memory Networks[http://arxiv.org/pdf/1503.08895v5.pdf]
	- The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations[http://arxiv.org/abs/1511.02301]

Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [NLTK](http://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/index.html)


Usage
-----

First, you need to download [DeepMind Q&A Dataset](https://github.com/deepmind/rc-data) from [here](http://cs.nyu.edu/~kcho/DMQA/), save `cnn.tgz` and `dailymail.tgz` into the repo, and run:

    $ ./unzip.sh cnn.tgz dailymail.tgz

Then run the pre-processing code with:

    $ python data_utils.py data cnn

To train a model with `cnn` dataset:

    $ python main.py --dataset cnn

To test an existing model:

    $ python main.py --dataset cnn --forward_only True


Thanks for Author Taehoon Kim / [@carpedm20](http://carpedm20.github.io/) provide our firest version tensorflow code


author: wujs(https://github.com/wujsAct)
