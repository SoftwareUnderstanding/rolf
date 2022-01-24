# PointNet - Ein Implementationsbeispiel mit Jupyter Notebooks

Dieses Projekt dient dazu, die von Qi et al. [1] entworfene PointNet-Architektur anhand einer interaktiven Beispiel-Implementation zu erläutern und somit einem breiteren Publikum verständlich zu machen.

Der Code dieses Projekt basiert auf https://github.com/garyli1019/pointnet-keras

Die Original-Implementation der Autoren ist in folgendem Repo zu finden: https://github.com/charlesq34/pointnet

## Installation

Via Anaconda:

- [**Anaconda**](https://www.anaconda.com/products/individual) installieren 

- Tensorflow und Keras via Anaconda Navigator zur root-Umgebung hinzufügen

- Server via Jupyter Notebook App starten

Via Pip:

- falls eine GPU von Nvidia genutzt wird: [**CUDA**](https://developer.nvidia.com/cuda-downloads) installieren

- [**Python 3.6.1 oder höher**](https://www.python.org/downloads/) installieren

- in der Kommandozeile zum Verzeichnis des Repositories navigieren und folgenden Befehl eingeben: "pip install -r requirements.txt"

- anschließend Jupyter Notebook über die Kommandozeile mit "jupyter notebook" starten

## Nutzung

PointNet mit geänderten Paramtern neu trainieren: Kernel => Restart & Run all

Um andere Segmentierungsbeispiele zu rendern, muss lediglich die letzte Codezelle nach Modifikation via Run erneut ausgeführt werden.

## Lizenz

Dieses Projekt wird unter der MIT-Lizenz veröffentlicht (siehe LICENSE-Datei)

## Quellen

[1] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep Learning on Point Sets for
3D Classification and Segmentation,” CoRR, vol. abs/1612.00593, 2016, [Online]. Available:
http://arxiv.org/abs/1612.00593.