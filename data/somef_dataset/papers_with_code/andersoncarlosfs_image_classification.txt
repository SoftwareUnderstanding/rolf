# Introduction 
Prenez un data scientist qui aurait conçu un algorithme de classification de données avec des résultats corrects. Demandez-lui comment l’améliorer. Neuf fois sur dix, vous vous entendrez répondre : “Il me faut plus de données !”. 

En général, avoir plus de données permet de créer des modèles plus proches de la réalité, qui se généraliseront mieux aux nouvelles données (qui n'ont pas servi à l'apprentissage) et permettront donc d’obtenir de meilleurs résultats. Mais le problème c’est que pour réaliser l’apprentissage de modèles sur de gros volumes de données, il faut mettre en place une architecture de stockage et de calcul appropriée.

# Getting Started
Le rôle d’un classifieur d’images est d’indiquer de manière automatique quel est le contenu principal d’une image, à partir d’une liste de classes possibles. 

Nous allons commencer par télécharger les images d'u dataset  qui comprend 7390 images en provenance de 37 classes différentes, chacune de ces classes correspondant à une race différente de chien et de chat ([Oxford IIIT-Pet Dataset]('http://www.robots.ox.ac.uk/~vgg/data/pets/')). Puis nous allons scinder nos données en deux groupes. Par exemple, pour chacune des 37 classes, nous allons mettre les 100 premières dans le jeu de données d'apprentissage (soit 3700 images en tout). Les 3690 autres images seront dans le jeu de données de test.

Pour chacune des images, il faut d’abord produire une représentation de l’image sous la forme d’un tableau de valeurs flottantes, ce qui permettra de réaliser des calculs sur cette représentation. Pour calculer cette représentation, nous allons utiliser l'état de l'art du domaine à savoir les réseaux de neurones convolutionnels (CNN) sous la forme d'une architecture profonde (Deep Learning). Il faudra pour chaque image appliquer un CNN et extraire les valeurs de sortie d’une des couches intermédiaires. 

En gros, on va obtenir une représentation d'images sous la forme d'un tableau de 4096 valeurs flottantes. C'est ce qu'on appelle des features. Ces features proviennent de l'avant-dernière couche de notre réseau de neurones.

Voilà un script Python qui permet d'extraire les features des images passées en argument et de sauvegarder les features au format JSON :

```python
#! /usr/bin/env python
import json
import sys

# Dependencies can be installed by running:
# pip install keras tensorflow h5py pillow

# Run script as:
# ./extract-features.py images/*.jpg

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def main():
    # Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    # This is going to take some time...
    base_model = VGG16(weights='imagenet')
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    # (see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    # For each image, extract the representation
    for image_path in sys.argv[1:]:
        features = extract_features(model, image_path)
        with open(image_path + ".json", "w") as out:
            json.dump(features, out)

def extract_features(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]

if __name__ == "__main__":
    main()
``` 

# Build and Test
A partir des features de toutes les images du jeu de données d’apprentissage, il faut réaliser l’apprentissage du modèle de classification. Ceci sera réalisé à l’aide de la fonction [SVMWithSGD]('https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.classification.SVMWithSGD') (Support Vector Machine with Stochastic Gradient Descent). 

Une fois le modèle créé, on l'applique aux features des images de test. Le modèle permet de prédire pour chaque image une classe, qui est correcte ou incorrecte. Le taux de succès du modèle est le pourcentage de bonnes classifications.

# Contribute
Votre mission est de créer une application Spark permettant de séparer les images en jeux d'apprentissage et de test, d'extraire les features des images en provenance de deux classes différentes, d'apprendre un modèle sur les données d'apprentissage et de mesurer les performances du modèle sur les données de test. À titre d'exemple, avec 100 images d'apprentissage dans chaque classe, il est possible d'obtenir 98% de bonnes classifications dans la tâche Wheaten Terrier vs Yorkshire Terrier.

Notez que vous pouvez apprendre un modèle "Classe X vs Classe Y" pour toutes les paires de classes (X, Y) possibles : c'est ce qu'on appelle la classification "1 vs 1". Une alternative est de créer un modèle "Class X vs Toutes les autres classes" : C'est ce qu'on appelle la classification "1 vs All". La différence est que dans la classification 1 vs 1, on part du principe que l'on sait déjà que les images appartiennent à la classe X ou Y : il n'y a pas d'images qui appartiennent à une classe autre que X ou Y. Alors qu'un modèle 1 vs All permet, à partir de n'importe quelle image de test, de prédire si elle appartient à la classe X ou non. Je vous conseille de commencer par la classification 1 vs 1, puis d'adapter votre programme pour faire du 1 vs All.

Une fois que vous aurez prototypé votre application en local, vous pourriez déployer un cluster de calcul sur Amazon Web Services qui vous permettra de réaliser l'apprentissage complet de vos modèles. D'ailleurs, les performances de classification pourront être récoltées dans S3.

Vous pourriez également détecter les goulots d'étranglement de votre application et,si possible, proposer des solutions permettant de s'en affranchir. 

# Copyrights
- OpenClassrooms

# Support me
- Binance Pay: [andersoncarlosfs](https://app.binance.com/cn/qr/dplk69e279fff5e8445ea2060689c0d56291) / [151298424](https://app.binance.com/cn/qr/dplk69e279fff5e8445ea2060689c0d56291) / [QR Code](https://raw.githubusercontent.com/andersoncarlosfs/resume/main/assets/images/binance_pay.jpeg)
- Binance P2P: andersoncarlosfs / [QR Code](https://raw.githubusercontent.com/andersoncarlosfs/resume/main/assets/images/binance_p2p.jpeg)
- Revolut: [andersoncarlosfs](https://revolut.me/andersoncarlosfs) 
