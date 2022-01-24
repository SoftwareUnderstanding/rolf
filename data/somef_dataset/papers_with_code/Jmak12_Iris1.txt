# Reconnaissance de l'IRIS
Du début à la fin, le projet utilise PyTorch.


## Installation

* Préparez les outils pour configurer l'environnement virtuel (si vous l'avez déjà fait, ignorez-le):
```
sudo apt-get install -y python-pip python3-pip cmake
mkdir ~/.virtualenvs
cd ~/.virtualenvs
sudo pip install virtualenv virtualenvwrapper
sudo pip3 install virtualenv virtualenvwrapper
echo "# virtualenv and virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
```

* Créez un nouvel environnement virtuel, nommé * iris *:
```
virtualenv -p python3.6 iris
workon iris
```

* Cloner git et installer les paquets requis: 
```
cd Iris-Recognition-PyTorch
pip install -r requirements.txt
pip install git+https://github.com/Jmak12/pytorch-image-models##egg=timm
```


## Entraînement
* Les jeux de données utilisés sont [MMU2] (https://www.cs.princeton.edu/~andyz/irisrecognition) et [CASIA1] (https://github.com/thuyngch/Iris-Recognition/tree/master/CASIA1). . Chaque jeu de données est utilisé indépendamment.

* Le jeu de données MMU2 contient 995 images correspondant à 100 personnes. Chaque personne a été capturée deux yeux et chaque œil a été capturé 5 images (Mais une personne n'est capturée qu'un œil). Dans lequel, avec chaque œil, 3 images aléatoires sont sélectionnées pour le kit d’entraînement et le reste appartient au groupe de test. Au total, la taille de la formation est 597 et la taille du test est 398. Vous trouverez la liste des ensembles de formation et la taille du test dans [data / mmu2_train.txt] (data / mmu2_train.txt) et [data / mmu2_valid.txt] (data /mmu2_valid.txt)

* Parallèlement, le jeu de données CASIA1 comprend 756 images correspondant à 108 personnes. Chaque personne a également été capturée deux yeux, celui de gauche capturé 3 fois et celui de droite capturé 4 fois. Je sélectionne au hasard 2/3 images de l'œil gauche et 2/2 images de l'œil droit pour l'ensemble d'entraînement. En résumé, la taille de la formation est de 432 et la taille du test est de 324. Vous trouverez la liste des ensembles de formation et la taille du test dans [data / casia1_train.txt] (data / casia1_train.txt) et [data / casia1_valid.txt] ( data / casia1_valid.txt)

|        | Train images | Test images |
|--------|--------------|-------------|
|  MMU2  |      597     |     398     |
| CASIA1 |      432     |     324     |

* En ce qui concerne le modèle, je me réfère à [EfficientNet] (https://arxiv.org/abs/1905.11946) avec deux variantes b0 et b1. Les modèles sont formés par l’optimiseur [SGDR] (https://arxiv.org/abs/1608.03983) avec 300 époques.

* Les fichiers de configuration peuvent être trouvés dans le dossier [configs] (configs). Pour démarrer le processus de formation à l'aide de EfficientNet-b0 sur le jeu de données MMU2 avec GPU0, utilisez la commande suivante:
```
python train.py --config config/mmu2_b0.json --device 0
```


## Resultats

* La perte et la précision sur le jeu de données MMU2 sont résumées et tracées comme suit:

|                 | Loss (train/valid) | Accuracy (train/valid) |
|-----------------|--------------------|------------------------|
| EfficientNet-b0 |    0.0105/0.0288   |      1.0000/0.9980     |
| EfficientNet-b1 |    0.0093/0.0202   |      1.0000/1.0000     |

<p align="center">
  <img src="pics/mmu2/b0_loss.png" width="430" alt="accessibility text">
  <img src="pics/mmu2/b0_acc.png" width="430" alt="accessibility text">
  <br>
  <em>Loss and Accuracy curve of EfficientNet-b0 on the MMU2 dataset</em>
</p>

<p align="center">
  <img src="pics/mmu2/b1_loss.png" width="430" alt="accessibility text">
  <img src="pics/mmu2/b1_acc.png" width="430" alt="accessibility text">
  <br>
  <em>Loss and Accuracy curve of EfficientNet-b1 on the MMU2 dataset</em>
</p>

* Loss and accuracy on the CASIA1 dataset are summarized and plotted as follows:

|                 | Loss (train/valid) | Accuracy (train/valid) |
|-----------------|--------------------|------------------------|
| EfficientNet-b0 |    0.0269/0.1179   |      1.0000/0.9742     |
| EfficientNet-b1 |    0.0152/0.1457   |      0.9980/0.9766     |

<p align="center">
  <img src="pics/casia1/b0_loss.png" width="430" alt="accessibility text">
  <img src="pics/casia1/b0_acc.png" width="430" alt="accessibility text">
  <br>
  <em>Loss and Accuracy curve of EfficientNet-b0 on the CASIA1 dataset</em>
</p>

<p align="center">
  <img src="pics/casia1/b1_loss.png" width="430" alt="accessibility text">
  <img src="pics/casia1/b1_acc.png" width="430" alt="accessibility text">
  <br>
  <em>Loss and Accuracy curve of EfficientNet-b1 on the CASIA1 dataset</em>
</p>

* Download trained weight:
```
gdown https://drive.google.com/uc?id=18-4JLAEJGa1D4My2mky4Co0WU1eDPq6X&export=download # mmu2_b0
gdown https://drive.google.com/uc?id=10sOieImsvre4msafbr07F_hdN_o6Pj0p&export=download # mmu2_b1
gdown https://drive.google.com/uc?id=1yAnpO_UotSP8zgVTGqOT0iLpGcAOF5EE&export=download # casia1_b0
gdown https://drive.google.com/uc?id=1VdAg-_Sjm3gVAg_KLpJkL6RktxAcgJZH&export=download # casia1_b1
```

* Pour que le modèle formé se concentre sur la région de l'iris à l'intérieur des images, j'utilise [Grad-CAM] (https://arxiv.org/abs/1610.02391) pour visualiser l'attention de la dernière couche d'entités (juste avant la mise en pool globale moyenne). Pour visualiser heatmap, utilisez cette commande:
```
python visualize.py --image /home/thuyngch/datasets/Iris/MMU2/010105.bmp \
                    --config config/mmu2_b0.json \
                    --weight /home/thuyngch/checkpoints/model_best.pth \
                    --use-cuda
```

<p align="center">
  <img src="pics/mmu2/b0_gradcam.jpg" width="500" alt="accessibility text">
</p>
