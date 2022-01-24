# CIFAR100 Image Classification

## Modules
```
|-- Data
|   |-- cifar-100-python
|-- Log
|-- Model
|   `-- resnet_cifar100.pt
|-- model.py
`-- train.py
```

## Model 
  1. ResNet 34-layers
  
## Model Specification
  1. Total layer size : 34
  2. Input image size : 224
  3. Input channel size : 64
  4. Input kernal size : 7
  5. Layer size list : [3, 4, 6, 3]
  6. Channel size list : [64, 128, 256, 512]
  7. Kernal size : 3
  8. Class size : 100

## Training 
  1. Optimizer : SGD (momentum = 0.9)
  2. Scheudler : StepLR
      * Init learning rate : 1e-2
      * Step Size : 5
      * Gamma = 0.5
  3. Epochs : 60
  4. Batch size : 64

## Data 
  1. CIFAR100

## Reference
  1. Resnet : https://arxiv.org/pdf/1512.03385.pdf

