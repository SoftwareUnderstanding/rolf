# Installation:

```
pip install -e .
```

# Installing DL2:

First clone DL2 in a separate directory and install it using the following commands:

```
git clone https://github.com/eth-sri/dl2
cd dl2
pip install -r requirements.txt
```

If you are using a virtual environment then make sure to install DL2 in that environment.
Now DL2 can be imported as a python libary.
To achieve this just extend the `PYTHONPATH` variable to also point to the DL2 directory:

```
export PYTHONPATH="${PYTHONPATH}:{path_to_dl2}"
```

# Execution:

Run CIFAR10 experiment:

```
run_image_experiments.py cifar10 --layers=10 --widen_factor=1 run
```

Run CIFAR100 experiment:

```
run_image_experiments.py cifar100 --layers=10 --widen_factor=1 run
```

# Generate experiment conditions:

```
cd scripts
python generate_experiments.py
```

# Help functions:

```
run_image_experiments.py -- --help
run_image_experiments.py cifar10 -- --help
run_image_experiments.py cifar100 -- --help
```

# Acknowledgements

- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- [Wide Residual Networks (WideResNets) in PyTorch](https://github.com/xternalz/WideResNet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
