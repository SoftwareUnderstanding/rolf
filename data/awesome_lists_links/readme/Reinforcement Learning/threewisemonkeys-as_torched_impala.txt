# Pytorch IMPALA

A Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (IMPALA) implemented in pytorch

## Requirements

1. [python 3.7+]()
```bash
$ sudo apt install python3.7
```

2. [pytorch](https://pytorch.org/)
```bash
$ pip install torch
```

3. [tensorboard](https://pytorch.org/docs/stable/tensorboard.html)
```bash
$ pip install tensorboard
```

## Usage
1. Edit hyperparameters in `main.py`

2. Train the model 
```bash
$ python train.py
```

3. Logs will be collected in specified folder. You can use `tensorboard` to view them in a browser
```bash
$ tensorboard --logdir ./logs/
```

4. Test the model
```bash
$ python test.py
```

Example 
```bash
$ python test.py -pp ./models/IMPALA_RacecarBulletEnv-v0_400.pt -hd 32 -en RacecarBulletEnv-v0 -ne 10 -el 1000 -ld ./logs/
```



## References 
1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures by Espeholt, Soyer, Munos et al.] (https://arxiv.org/pdf/1802.01561.pdf)

## TODO
- [X] Fix OSError
- [X] Add batched updates
- [X] Add tensorboard logging
- [X] Test performance
- [ ] Fix performance collapse issues
- [ ] Add comand line argument support

