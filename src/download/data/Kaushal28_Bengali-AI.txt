# [Bengali.AI](https://www.kaggle.com/c/bengaliai-cv19): Different approaches

## Experiment #1 (0.9619 on public LB):

- Used Model: Pretrained EfficientNet-B1
- Image Augmentation technique(s): GridMask
- Other settings:
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 30 epochs for each fold
  - Batch size: 256

TODO:
  - ~~Commit Pretrained Models~~ Link to Kaggle dataset: https://www.kaggle.com/kaushal2896/trainedmodeleffnet
  
## Experiment #2 (0.9528 on public LB):

- Used Model: Pretrained EfficientNet-B1
- Image Augmentation technique(s): None
- Other settings:
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 256
  
TODO:
  - ~~Commit Pretrained Models~~ Link to Kaggle dataset: https://www.kaggle.com/kaushal2896/efficientnetnoaug
  
## Experiment #3: (0.9547 on public LB)

- Used Model: Pretrained EfficientNet-B1
- Image Augmentation technique(s): GridMask + AugMix
- Other settings:
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 256
  
TODO:
  - ~~Commit Pretrained Models~~ Link to kaggle dataset: https://www.kaggle.com/kaushal2896/efficientnet40epochsgridmaskaugmix
  
## Experiment #4: (0.9650 on public LB)

- Used Model: Pretrained ResNet34
- Image Augmentation technique(s): GridMask
- Other settings:
  - Image size 137 x 236 (original image size)
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 256
  
TODO:
  - ~~Commit Pretrained Models~~ Link to kaggle dataset: https://www.kaggle.com/kaushal2896/resnet34originalsize
  
 ## Experiment #5: (0.9676 on public LB)

- Used Model: Pretrained EfficientNet-B3
- Image Augmentation technique(s): GridMask
- Other settings:
  - Image size 137 x 236 (original image size)
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 128
  
TODO:
  - ~~Commit Pretrained Models~~ Link to Kaggle dataset: https://www.kaggle.com/kaushal2896/bengaliaieffnetb3
 
 ## Experiment #6: (0.9692 on public LB)

- Used Model: Pretrained EfficientNet-B3 + Pretrained ResNet-34 \[Ensemble\]
- Image Augmentation technique(s): GridMask
- Other settings:
  - Image size 137 x 236 (original image size)
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 128
  
 ## Experiment #7: (0.9644 on public LB)

- Used Model: Pretrained EfficientNet-B3
- Image Augmentation technique(s): CutMix(30%), MixUp(30%)
- Other settings:
  - Image size 137 x 236 (original image size)
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 128
  
 ## Experiment #8: 

- Used Model: Pretrained EfficientNet-B3 + Pretrained ResNet-34 \[Ensemble\]
- Image Augmentation technique(s): GridMask + CutMix + MixUp
- Other settings:
  - Image size 137 x 236 (original image size)
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 128

References:
  - https://www.youtube.com/watch?v=8J5Q4mEzRtY (Part1 and Part2)
  - https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
  - GridMask: https://arxiv.org/pdf/2001.04086.pdf, https://www.kaggle.com/haqishen/gridmask
  - AugMix: https://arxiv.org/pdf/1912.02781.pdf, https://www.kaggle.com/haqishen/augmix-based-on-albumentations
