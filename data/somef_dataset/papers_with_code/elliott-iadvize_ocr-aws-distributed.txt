# OCR#1 : Réalisez un apprentissage distribué - [~](https://github.com/elliott-iadvize/ocr-aws-distributed "~")

## Français

### Introduction

L'objectif du projet est de créer une application Spark permettant de séparer les images en jeux d'apprentissage et de test, d'extraire les features des images en provenance de deux classes différentes, d'apprendre un modèle sur les données d'apprentissage et de mesurer les performances du modèle sur les données de test.

> À titre d'exemple, il nous est indiqué qu'avec 100 images d'apprentissage dans chaque classe, il est possible d'obtenir 98% de bonnes classifications dans la tâche Wheaten Terrier vs Yorkshire Terrier.


#### Laïus sur la conception de l'application

Dans l'introduction au projet, il est conseillé de démarrer la construction de l'application par la classification 1 vs 1, ce qui a été fait ici, cependant j'ai préféré simplifier le code de l'application finale en n'y laissant que ce qui était relatif à la classification 1 vs All étant donné que c'est ce qui a été déployé et distribué.

En commençant par la classification 1 vs 1, j'ai pu plus facilement itérer sur la conception de l'application, j'ai aussi limité dans un premier temps le volume d'images et donc de features à exploiter pour accélérer le traitement étant donné qu'il s'agissait de tester la logique dans un premier temps. J'ai ensuite adapté le fonctionnement pour faire de la classification 1 vs All en appliquant la même stratégie de limitation des volumes.

Une fois l'application prototypé en local, il a été possible de la déployer sur un cluster de calcul AWS afin de réaliser l'apprentissage complet des modèles pour ensuite passer à une phase d'analyse des résultats puis d'optimisation des performances.

### Instructions

Ces instructions vous permettront de faire fonctionner l'application sur un cluster AWS EMR.

#### Pré-requis

Définition de variable dans le cadre de ces instructions:
`${Bucket Name} -> le nom du bucket AWS dans lequel les features sont stockées, par exemple : s3://hodor-loves-data`

Un script permettant d'extraire les features à partir des images récupérées ici [Oxford IIIT-Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/ "Oxford IIIT-Pet Dataset") est fourni dans le cadre du projet :

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import sys

#Dependencies can be installed by running:
#pip install keras tensorflow h5py pillow

#Run script as:
#./extract-features.py images/*.jpg

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def main():
    #Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    #This is going to take some time...
    base_model = VGG16(weights='imagenet')
    #Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    #(see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    #For each image, extract the representation
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

Pour chaque image présente dans le dossier en argument, le script va produire un fichier JSON contenant une représentation de l'image sous la forme d’un tableau de valeurs flottantes, ce qui permettra de réaliser des calculs.

Dans le cadre du projet, j'ai réalisé l'extraction des features une seule fois pour éviter de l'intégrer à l'application et donc réaliser ce traitement à chaque itération de la phase de développement. Au moment du déploiement sur le cluster, j'ai uploadé les features dans le bucket S3 en m'assurant que les permissions adéquates étaient en place. Je considère cela comme un pré-requis dans le cadre de ce readme de la même façon qu'il sera nécessaire d'uploader l'application.

Pour l'extraction des features, j'ai utilisé la version `3.6.5` de Python et voici la version des dépendances utilisées pendant la phase de développement :
```python
#requirements.txt

keras==2.2.4
tensorflow==1.12.0
h5py==2.8.0
pillow==5.3.0
boto3==1.7.4
```

#### Configuration du cluster et déploiement

Dans le cadre de la configuration du cluster de calculs, il m'a été nécessaire d'uploader un script bash sur AWS S3 qui est exécuté au moment de l'initialisation du cluster de façon à installer la librairie `boto3` qui permet de lire dans le bucket S3.   

```bash
#!/bin/bash
sudo pip install -U 'boto3==1.7.4' --force-reinstall
```

Un second fichier est aussi nécessaire en local, il s'agit d'un fichier de configuration Spark utilisé lui aussi au moment de l'initiatlisation du cluster de façon à allouer 2 Go de memoire par executor étant donné que j'ai rencontré de nombreux problème de mémoire avec erreur OOM, cette configuration a permis de résoudre le soucis :

```json
[
  {
    "Classification": "spark",
    "Properties": {
      "maximizeResourceAllocation": "true"
    }
  },
  {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.executor.memory": "2G"
      }
    }
]
```

Pour créer le cluster, j'utilise le CLI mis à disposition de façon à simplifier l'itération et la phase de déploiement, voici la commande finale:

```bash
aws emr create-cluster
  --applications Name=Ganglia Name=Spark Name=Zeppelin
  --ec2-attributes '{"KeyName":"iadvize_eau_ocr","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-278fcb4e","EmrManagedSlaveSecurityGroup":"sg-02b2a819721044d5d","EmrManagedMasterSecurityGroup":"sg-016b7b7113298da7f"}'
  --service-role EMR_DefaultRole
  --enable-debugging
  --release-label emr-5.20.0
  --log-uri 's3n://aws-logs-370616375808-eu-west-3/elasticmapreduce/'
  --name 'Cluster OCR'
  --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master Instance Group"},{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core Instance Group"}]'
  --configurations '[{"Classification":"spark","Properties":{},"Configurations":[]}]'
  --scale-down-behavior TERMINATE_AT_TASK_COMPLETION
  --region eu-west-3
  --bootstrap-actions Path=${Bucket Name}/install_dependencies.sh  
  --configurations file://./spark_config.json
```

#### Connexion au cluster, exécution de l'application et observation des résultats

Après avoir suivi les instructions de connexion en SSH au cluster fournies par AWS il est possible de se connecter au cluster EMR qui a préalablement été configuré en accord avec nos besoins et de lancer notre application Spark via la commande suivante:

```bash
sudo spark-submit --deploy-mode cluster ${Bucket Name}/classifier.py ${Bucket Name}
```

*Dans mon cas j'ai préféré créer un dossier `images/` dans mon bucket contenant toutes les features cependant il est tout à fait possible d'adapter le script.*

Une fois le script en cours, les logs apparaitront dans la console et il est aussi possible d'aller observer l'exécution de l'application Spark à partir des interfaces dédiées qu'AWS rend disponible qui seront aussi nécessaire à l'optimisation de l'application.

## English

### Introduction

The purpose of the project is to create a Spark application to separate images into learning and test sets, extract features from images from two different classes, learn a model using the learning data and measure the model's performance on the test data.

> For example, we are told that with 100 learning images in each class, it is possible to obtain 98% good classifications in the Wheaten Terrier vs Yorkshire Terrier task.

#### Laius on the design of the application

In the introduction to the project, it is recommended to start building the application with 1 vs 1 classification, which has been done here, however I preferred to simplify the code of the final application by leaving only what was related to 1 vs All classification since it is what has been deployed and distributed.

Starting with 1 vs 1 classification, I was able to iterate more easily on the design of the application, I also limited the volume of images and therefore features to use to speed up processing since it was a matter of testing the logic in the first place. I then adapted the operation to make the 1 vs All classification by applying the same volume limitation strategy.

Once the application was prototyped locally, it was possible to deploy it on an AWS computing cluster in order to complete the full learning of the models and then move on to a phase of results analysis and performance optimization.

### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Variable definition in the context of these instructions:
`${Bucket Name} -> name of the bucket on which features are stored, for instance: s3://hodor-loves-data`

A script to extract features from the images retrieved here [Oxford IIIT-Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/ "Oxford IIIT-Pet Dataset") is provided as part of the project:

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import sys

#Dependencies can be installed by running:
#pip install keras tensorflow h5py pillow

#Run script as:
#./extract-features.py images/*.jpg

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def main():
    #Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    #This is going to take some time...
    base_model = VGG16(weights='imagenet')
    #Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    #(see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    #For each image, extract the representation
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

For each image in the argument folder, the script will produce a JSON file containing a representation of the image in the form of a floating value table, which will allow calculations to be performed.

As part of the project, I extracted the features only once to avoid integrating them into the application and therefore performing this processing at each iteration of the development phase. When I deployed on the cluster, I uploaded the features into the S3 bucket making sure that the appropriate permissions were in place. I consider this a prerequisite for this readme.

For feature extraction, I used the `3.6.5` version of Python and here is the version of the dependencies used during the development phase:
```python
#requirements.txt

keras==2.2.4
tensorflow==1.12.0
h5py==2.8.0
pillow==5.3.0
boto3==1.7.4
```
#### Cluster configuration and deployment

As part of the configuration of the EMR cluster, it was necessary for me to upload a bash script on AWS S3 which is executed at the time of the initialization of the cluster in order to install the `boto3` library which allows to read in the S3 bucket.

```bash
#!/bin/bash
sudo pip install -U'boto3===1.7.4' --force-reinstall
```

A second file is also required locally, it is a Spark configuration file also used at the time of the cluster initialization in order to allocate 2 GB of memory per executor since I have encountered many memory problems with OOM error, this configuration has solved the problem:

```json
[
  {
    "Classification": "spark",
    "Properties": {
      "maximizeResourceAllocation": "true"
    }
  },
  {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.executor.memory": "2G"
      }
    }
]
```

To create the cluster, I use the CLI provided to simplify the iteration and deployment phase, here is the final command:

```bash
aws emr create-cluster --applications Name=Ganglia Name=Spark Name=Zeppelin --ec2-attributes '{"KeyName":"iadvize_eau_ocr","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-278fcb4e","EmrManagedSlaveSecurityGroup":"sg-02b2a819721044d5d","EmrManagedMasterSecurityGroup":"sg-016b7b7113298da7f"}' --service-role EMR_DefaultRole --enable-debugging --release-label emr-5.20.0 --log-uri 's3n://aws-logs-370616375808-eu-west-3/elasticmapreduce/' --name 'Cluster OCR' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master Instance Group"},{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":1}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core Instance Group"}]' --configurations '[{"Classification":"spark","Properties":{},"Configurations":[]}]' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region eu-west-3 --bootstrap-actions Path=${Bucket Name}/install_dependencies.sh  --configurations file://./spark_config.json
```

#### Connection to the cluster, execution of the application and observation of the results

After following the instructions provided by AWS to connect in SSH to the cluster it is possible to connect to the EMR cluster that has previously been configured according to our needs and launch our Spark application using the following command:

```bash
sudo spark-submit --deploy-mode cluster ${Bucket Name}/classifier.py ${Bucket Name}
```

*In my case I preferred to create an `images/` folder in my bucket containing all the features however it is quite possible to adapt the script.*

Once the script is running, the logs will appear in the console and it is also possible to observe the execution of the Spark application from the dedicated interfaces that AWS makes available, those interface will also be necessary to optimize the application.
