Code to run hybrid sampling schemes to compare against the softmax sampling. All experiments are run on langauge modeling tasks. The datasets used are Penn Tree Bank, Wikitext 2 and Wikitext 103. We will just use a LSTM/biLSTM for all the experiments. The goal here is to prove the efficacy of these hybrid schemes and not the model itself. 
<br/>
Added: sampled softmax. This is a simple reimplementation of https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
<br/>
Added weight tying as in https://arxiv.org/pdf/1608.05859v3.pdf
<br/>
Added all the kernel methods except the clustering estimators. <br/>

The code needs cleanup. <br/>

Added Alias sampling methods. https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ <br/>



Implementation of https://papers.nips.cc/paper/2019/file/e43739bba7cdb577e9e3e4e42447f5a5-Paper.pdf <br/>

