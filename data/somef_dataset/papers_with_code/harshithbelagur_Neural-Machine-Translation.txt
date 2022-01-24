# Neural Machine Translation

## TensorFlow 2.0 implementation of the popular NLP paper by Bahdanau et al. - Neural Machine Translation by Jointly Learning to Align and Translate (ICLR, 2015)

Find the paper at https://arxiv.org/pdf/1409.0473.pdf

For detailed implementation with replicated results, use **NMT_6400000** 

Following are the specifications followed as per the authors: 
- AdaDelta Optimizer with epsilon = 10-6 , rho = 0.95 
- Minibatch SGD with batch_size = 80 
- Embedding Dimension = 620 
- Hidden Layer Size = 1000
- Output Layer Size = 500 
- Weights initialization = RandomNormal with Mean = 0 and Standard Deviation = 0.001 
- Bias initialization = Zeros
- L2 Regularization

Total number of parameters = 28,332,000 (Encoder) + 3,003,001 (Attention) + 52,496,000 (Decoder)
                           = **90,831,001**
                           
 Use Presentation.pdf for a better understanding.
