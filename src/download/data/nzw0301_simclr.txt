# SimCLR re-implementation

PyTorch's re-implementation of SimCLR on CIFAR-10/100 on multiple GPUs.

## Requirements

- [PyTorch](https://pytorch.org/) >= 1.5.0
- [Hydra](https://hydra.cc/) >= 1.0.0
- [Apex](https://github.com/NVIDIA/apex)

---

## Run on 4 GPUs on a single machine

### Representation learning

```bash
python launch.py --nproc_per_node=4 -m main parameter.epochs=200
python launch.py --nproc_per_node=4 -m main parameter.epochs=1000  # longer training
```

### Supervised learning

The default classifier is centroid classifier.

```bash
python -m eval experiment.target_dir=PATH_TO_TRAINED_WEIGHTS
```

#### Linear and nonlinear classifiers with softmax

```bash
python -m eval parameter.classifier=linear experiment.target_dir=PATH_TO_TRAINED_WEIGHTS
python -m eval parameter.classifier=nonlinear experiment.target_dir=PATH_TO_TRAINED_WEIGHTS
```

### Results of CIFAR-10

Note: The reported accuracies are the __best validation__ accuracies.

#### epochs = 200

| Classifier | With projection head   | Without projection head |
|------------|-------:|-----------:|
| Centroid   | 0.7328 | 0.5271 |
| Linear     | 0.7868 | 0.8225 |
| Nonlinear  | 0.7946 | 0.8370 |


#### epochs = 1000

| Classifier | With projection head   | Without projection head |
|------------|-------:|-----------:|
| Centroid   | 0.8440 | 0.8442 |
| Linear     | 0.8665 | 0.8937 |
| Nonlinear  | 0.8759 | 0.8941 |

---

### (Traditional) Supervised performance

```bash
python launch.py --nproc_per_node=4 -m supervised parameter.epochs=200
```

The best validation accuracy is `92.75%`.

---

## References

- Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. __[A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.icml.cc/static/paper_files/icml/2020/6165-Paper.pdf)__, In _ICML_, 2020.
