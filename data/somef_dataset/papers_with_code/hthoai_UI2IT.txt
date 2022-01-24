<div align="center">

# UI2IT
**Reimplement Unsupervised Image-to-Image Translation Networks (PyTorch)**

</div>

## Introduction
- Replaced Visdom with TensorBoard.
- Restructured in a clear way.
- Less GPU usage than the original code.

```.
.
├── config              # Model config
├── datasets            # Data location
├── experiments         # Log, save model...
├── lib
│   ├── datasets        # Dataset classes
│   ├── layers          # Layer classes
│   ├── models          # Model classes
│   ├── config.py       # Class to load config
│   ├── experiment.py   # Experiment class, define callback here
│   └── runner.py       # Train/val loop
├── tensorboard
├── utils
├── main.py
└── README.md
```

## Usage

```bash
python main.py train --exp_name <experiment_name> --cfg <config_path>
```

## Roadmap

| Method | Venue | Status |
|:-------|:------|:-------|
| [CycleGAN](https://arxiv.org/abs/1703.10593) | ICCV 2017 | Done |
| [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848) | NIPS 2017 | In progress |
| [Contrastive Learning for Unpaired Image-to-Image Translation](https://arxiv.org/abs/2007.15651) | ECCV 2020 | To-do |
| [Dual Contrastive Learning for Unsupervised Image-to-Image Translation](https://arxiv.org/abs/2104.07689) | CVPRW 2021 | To-do |