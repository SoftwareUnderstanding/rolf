# coral-net

Coral image dataset.

## Why?
I was looking for a dataset that would be hard for the model to train on - something similar to ImageWoof https://github.com/fastai/imagenette but larger in size, with large class imbalances, something that would be a challenge.
## Dataset
The dataset was created by web scraping images from Facebook, Instagram, Google Images, Duck Duck Go Images and various coral seller pages (WWC, White Corals, Euro Corals etc.) - around 30 different sources. 

Final dataset consisted of around 200K unlabeled images. I had manually labeled them into 37 classes (initially, around 100 images per class) described below:

    'Acanthastrea',
    'Acanthophyllia & Cynarnia',
    'Acropora',
    'Alveopora & Goniopora',
    'Blastomussa',
    'Bubble Coral',
    'Bubble Tip Anemone',
    'Candy Cane Coral',
    'Carpet Anemone',
    'Chalice',
    'Cyphastrea',
    'Discosoma Mushroom',
    'Elegance Coral',
    'Euphyllia',
    'Favia',
    'Gorgonia',
    'Leptastrea',
    'Leptoseris',
    'Lobophyllia & Trachyphyllia & Wellsophyllia',
    'Maze Brain Coral Platygyra',
    'Mini Carpet Anemone',
    'Montipora',
    'Pavona',
    'Plate Coral Fungia',
    'Porites',
    'Psammacora',
    'Rhodactis Mushroom',
    'Ricordea Mushroom',
    'Rock Flower Anemone',
    'Scolymia',
    'Scroll Corals Turbinaria',
    'Star Polyps',
    'Styllopora & Pocillipora & Seriatopora',
    'Sun Corals',
    'Toadstool & Leather Coral',
    'Tridacna Clams',
    'Zoa'

Sample images:
![samples](./images/sample-images.png)


Some of the classes consist of couple of different genera since to distinguish between them takes a domain expert - which I am not...

Following Data preprocessing steps have been applied:
* Rename all images into following format `<guid with no "-" signs>.<extension>`
* Convert all images to `*.jpg`
* Duplicate image removal using CNN method from `imagededup` package with 95% similarity threshold

After doing the training loop for couple of times I've managed to build a fairly clean dataset of **130K images, 4.3GB in size**. The rest of them are still unlabeled. The model tends to give high confidence to images that do not represent a coral (images of people, equipment, text etc), possible tweaks are needed...

![class counts](./images/img-instances-per-class.png)

Due to possible legal stuff I am not providing a direct link for the dataset. 

## Training loop
Once every class had at least 100 images I trained a simple `resnet34` ensemble to get better results.

* 10-k Fold CV was used
* I've tried to use bigger models like `resnext50_32x4d`, but the results did not improve in any meaningful way, and training time increased significantly (even with mixed precision training)
* Mixed Precision slightly improved the results and significantly improved training time
* Inference time decision was done using voting:
    * If all classifiers voted unanimously then no soft voting was performed
    * Otherwise soft voting was performed and decision threshold was set to 90%
* After entire dataset was assigned weak labels, I manually went through the images and applied corrections if necessary 
* Weakly labeled images were then added to the training set and the whole process was repeated

This process is called Human-in-the-loop training. Other approach would be to use something called Self-training with Noisy Student https://arxiv.org/abs/1911.04252.
But that requires way bigger dataset.
 
## Tricks and tools
* Pytorch
* One Cycle LR from Super-Convergence paper https://arxiv.org/abs/1708.07120
* Imbalanced dataset sampler based on https://github.com/ufoym/imbalanced-dataset-sampler
* Pytorch-Lightning https://pytorch-lightning.readthedocs.io/en/latest/
* Tensorboard integration for training process monitoring and logging the Confusion Matrices
* Transfer learning (no splits were used, for some reason this worked slightly better than training different param-groups separately)
* Human-in-the-loop training
* 10 Fold CV
* Duplicates removal with CNN https://github.com/idealo/imagededup
* Dask for quick file renaming
* Gradient Accumulation
* Early stopping
* LR Logger


## Results
The single best model reached **F1 Score of 0.7474** (which was my target metric).
Other metrics for this model were as follow:

### Metrics
| Metric Name | Metric Value |
|--|--|
| **F1-Score** | **0.7474** |
| Overall Accuracy | 0.9561 |
| Top 1 Accuracy | 0.9789 |
| Top 5 Accuracy | 0.9971 |
| Precision | 0.7483 |
| Recall | 0.7548 |
| Loss | 0.1055 |

### Confusion Matrix
Confusion matrix on last training loop:
![Confusion matrix](./images/confusion-matrix.png)

### Logs
Validation logs:
![Val logs](./images/val-logs.png)
