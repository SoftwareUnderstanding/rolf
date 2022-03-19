# Neural-Turing-machine
NTM in PyTorch<br>
https://arxiv.org/pdf/1410.5401.pdf <br>

### How to use

```
usage: train.py [-args]

optional arguments:
  -h, --help              show this help message and exit
  --sequence_length       The length of the sequence to copy (default: 3)
  --token_size            The size of the tokens making the sequence (default: 10)
  --memory_capacity       Number of records that can be stored in memory (default: 64)
  --memory_vector_size    Dimensionality of records stored in memory (default: 128)
  --training_samples      Number of training samples (default: 999999)
  --controller_output_dim Dimensionality of the feature vector produced by the controller (default: 256)
  --controller_hidden_dim Dimensionality of the hidden layer of the controller (default: 512)
  --learning_rate         Optimizer learning rate (default: 0.0001)
  --min_grad              Minimum value of gradient clipping (default: -10.0)
  --max_grad              Maximum value of gradient clipping (default: 10.0)
  --logdir                The directory where to store logs (default: logs)
  --loadmodel             The pre-trained model checkpoint (default: '')
  --savemodel             Path/name to save model checkpoint (default: 'checkpoint.model')
```

## Copy task: <br>
<img src="./img/output.jpg" width="600">

### Memory snapshot: <br>
<img src="./img/memory.jpg" width="300">
