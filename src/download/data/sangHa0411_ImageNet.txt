# ImageNet Classification

## Model 
  1. DenseNet 169
  
## Model Specification
  1. Input image size : 224
  2. Input kernal size : 7
  3. Layer size list : [6, 12, 32, 32]
  4. Growth Rate : 24
  5. Kernal size : 3
  6. Class size : 1000

## Data Augmentation
  1. Train
      > 1) Resize (size : 256)
      > 2) RandomHorizontalFlip
      > 3) RandomCrop(size : 224)
      > 4) Normalize
      > 5) CutMix

  2. Test
      > 1) Resize (size : 224)
      > 2) Normalize

## Training 
  1. Optimizer : SGD (momentum = 0.9)
  2. Scheudler : StepLR
      * Init learning rate : 1e-2
      * Step Size : 10
      * Gamma = 0.5
  3. Epochs : 100
  4. Batch size : 64

## Data 
  1. ImageNet
      1. Train Data Size : 1000000
      2. Val Data Size : 50000
      3. Class Size : 1000

## Reference
  1. Densenet : https://arxiv.org/pdf/1608.06993.pdf
  2. CutMix : https://arxiv.org/pdf/1905.04899.pdf

