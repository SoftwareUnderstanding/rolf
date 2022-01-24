# Transformer
 
A TensorFlow 2.x implementation of the Transformer from [`Attention Is All You Need`](https://arxiv.org/pdf/1706.03762.pdf) (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 
 
This is my attempt at trying to understand and recreate the transformer from the research paper. This is just for my own understanding of the subject and is by no means perfect. 

In order to understand and implement the transformer I've taken the help of various tutorials and code guides, which I'll be linking in the resources section.

## Requirements
- tensorflow==2.1.0
- numpy==1.16.5
- tensorflow_datasets==3.2.1

## How to run
`python train.py`
 
## Resources
- The original paper: https://arxiv.org/pdf/1706.03762.pdf
 - Input and training pipeline: https://www.tensorflow.org/tutorials/text/transformer
- An useful article explaining the paper: https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
- Another useful article explaining the paper: http://jalammar.github.io/illustrated-transformer/
- A tensorflow 1.x transformer implementation: https://github.com/Kyubyong/transformer/
- The official implementation in tensorflow: https://github.com/tensorflow/tensor2tensor
 
