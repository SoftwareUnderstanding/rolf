# Padam vs ATMO
This code is a fork of [Padam](https://github.com/uclaml/Padam) offical code to obtain a perfect comparison between [ATMO](https://gitlab.com/nicolalandro/multi_optimizer) idea and Padam.

## Prerequisites: 
```
pip install -r requirements.txt
```

## Usage:
Use python to run run_cnn_test_cifar10.py for experiments on [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) and run_cnn_test_cifar100.py for experiments on [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Command Line Arguments:
* --lr: (start) learning rate 
* --method: optimization method, e.g., "sgdm", "adam", "amsgrad", "padam", "mps", "mas", "map"
* --net: network architecture, e.g. ["vggnet"](https://arxiv.org/abs/1409.1556), ["resnet"](https://arxiv.org/abs/1512.03385), ["wideresnet"](https://arxiv.org/abs/1605.07146)
* --partial: partially adaptive parameter for Padam method
* --wd: weight decay
* --Nepoch: number of training epochs
* --resume: whether resume from previous training process

## Usage Examples:
* Run experiments on Cifar10:
```bash
python run_cnn_test_cifar10.py  --lr 0.01 --method "mps" --net "resnet"  --partial 0.125 --wd 2.5e-2 > logs/resnet/file.log
```
* Obtain max and mean of logs
```
python folder_mean_accuracy.py
```

## Results

SGD-Momentum | ADAM | Amsgrad | AdamW | Yogi | AdaBound | Padam | Dynamic ATMO
:----- | :----: | :----: | :----: | :----: | :----: | :----: | -----:
 95.00 | 92.89 | 93.53 | 94.56 | 93.92 | 94.16 | 94.94 | **95.27**

## Citation
Please check [our paper](https://www.mdpi.com/1999-4893/14/6/186) for technical details and full results. 

```
@article{
  title={Combining Optimization Methods Using an Adaptive Meta Optimizer},
  author={Nicola Landro and Ignazio Gallo and Riccardo La Grassa},
  year={2021},
  journal={Algorithms MDPI},
}
```
