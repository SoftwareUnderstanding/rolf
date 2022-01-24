# RegNet-pytorch
A pytorch implement of RegNet (Designing Netowrk design spaces). Original paper link: https://arxiv.org/pdf/2003.13678.pdf

The performance of this repository hasn't been tested because of lacking resource on the computation, which may update in the future.

# How to Use
## 1.Dataset
```
Prepare a train.txt(or a val.txt) file for training(testing) your custom dataset.
  train.txt is organized as:
    your/data/path/img_0.jpg  0(label of img_0.jpg)
    your/data/path/img_1.jpg  1
    ......
    
  The separator between img_path and its_label is '\t'
```

## 2.training
```
  1. Create a 'training.yml' file like 'AnyNet_cpu.yml' in 'Data' folder
  
  2. Open train.py and find:
    'if __name__=='__main__':'
    Change the the path for 'load_cfg' to your '.yml' file
    
  3. Run the train.py
```

## 3.test
```
  1. Prepare the '.yml' file at first
  
  2. Open test.py and find:
    'if __name__=='__main__':'
    Change the path for 'load_cfg' to your '.yml' file
    
  3. Run the test.py
```
# Reference
[facebookresearch/pycls](https://github.com/facebookresearch/pycls)
