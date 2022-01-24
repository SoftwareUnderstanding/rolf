# Neural Turing Machine Implementation in PyTorch

Implementation of a Neural Turing Machine from the Graves et al paper (2014) 
    
[https://arxiv.org/pdf/1410.5401.pdf](https://arxiv.org/pdf/1410.5401.pdf)

Example:

    $ python3 run.py <model_type> <task_type> > test_out.txt

Args:
  -  **model_type** string indicating the RNN model to use, options:
       - BasicRNN, BasicLSTM, and NTM_LSTM
    
  -  **task_type** string indicating the toy task to use, options:
       - Single, StartSequence3, StartSequence20
        
Returns:
    prints periodic outputs of the average loss rate over the
    last batch of examples run through the model, e.g.
    
    $ [0] Error: tensor(0.0022, device='cuda:0', grad_fn=<DivBackward0>)
    $ [1000] Error: tensor(1.9112, device='cuda:0', grad_fn=<DivBackward0>)
