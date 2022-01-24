# Style-Based GAN in PyTorch

Implementation of A Style-Based Generator Architecture for Generative Adversarial Networks (https://arxiv.org/abs/1812.04948) in PyTorch

## Dependencies

- Python 3.6.8
- PyTorch 1.3

## Dataset

[FFHQ](https://github.com/NVlabs/ffhq-dataset)

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Acknowledge

Most codes are borrowed from https://github.com/rosinality/style-based-gan-pytorch.