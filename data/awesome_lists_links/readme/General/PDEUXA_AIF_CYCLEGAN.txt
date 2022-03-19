# Réseaux adverses génératifs

![RAG](imgs/RAG.png?raw=true "RAG")

## Principe

Un RAG est un modèle génératif où deux réseaux sont placés en 
compétition.
Le premier réseau est le générateur, il génère un échantillon (ex. du bruit = pire cas)
, tandis que son adversaire, le discriminateur essaye de détecter si un échantillon est réel ou
 bien s'il est le résultat du générateur. 


![Principe RAG](imgs/RAGprincipe.png "Rag Principe") <!-- .element height="50%" width="50%" -->

## Cycle Gan
Le CycleGan est une technique de GAN qui permet d'entrainer
des modèles de génération d'image, de type image-to-image, sans exemples apairés.

Le CycleGan permet de passer d'une image à une autre via deux générateurs
différents.
 
## Examples
![Example cRAG](imgs/CycleGanExample.jpg "Example cRAG")


## Comment débuter ?

### -- Prérequis --

```
Installer les prérequis issu du fichier requirements.txt
Installer tensorflow > 2.0.0
```

### -- Installation --
<ul>
<p>
Docker
</p>
</ul>
Vous pouvez directement créer un environnement stable via docker
Depuis le répertoire principal, executer la commande

```
docker build -t cyclegan .
docker run -it --name cycleganC cyclegan
```
<hr>
<ul>
<p>
Déploiement cloud
</p>
</ul>

Vous pouvez suivre cette procedure pour déployer une machine virtuel sur Google Cloud
```
Créer une instance 'Deep Learning VM '
8vCPU, 30 Gb RAM, GPU NVIDIA P100(ou autre), avec le framework TensorFlow Enterprise 2.1 (CUDA 10.1).
Cocher la case 'Install NVIDIA GPU driver automatically on first startup?'
Copier votre clef publique dans la VM, afin de pouvoir y acceder en SSH.
Copier le contenu du projet "AIF_CycleGan" via un scp -r ou via un gitclone.
Mettre à jour les prérequis avec mise_a_jour.sh (optionnel sur une instance Deep Learning VM).
Lancer python3 main.py ou un tunneling SSH pour utiliser le notebook.
```
Tunneling SSH
```
Depuis la VM: 
jupyter notebook --no-browser --port=8080
Depuis l'ordinateur local
ssh -N -L 8080:localhost:8080 <IDuser>@<ipVM>
```

## Pour faire tourner le cycleGan
```
python3 main.py
```
Les arguments suivants peuvent être ajoutés.
```
'--dataset', default='ukiyoe2photo'
'--batch_size', type=int, default=1
'--epochs', type=int, default=50
'--cycle_loss_weight', type=float, default=10.0
'--identity_loss_weight', type=float, default=0
```
Le fichier main.py va générér les différents réseaux, et lancer la phase d'entrainement, des checkpoints sont créés toutes les 5 epochs afin de pouvoir fractionner la phase d'entrainement.
<ul>
<li>checkpoints: Répétoire de sauvegarde des checkpoints d'entrainement</li>
<li>data: Répétoire regroupant les fonction de traitement des donénes (chargement, transformation, affichage)</li>
<li>model: Répétoire regroupant les fonctions relatifs aux modèles</li>
<li>Dockerfile: Création du docker</li>
<li>mise_a_jour.sh: Script d'installation des requirements.txt</li>
<li>requirements.txt: Ensemble des prérequies du projet</li>
<li>train.py: Fonction d'entrainement</li>
</ul>

## Quelques résultats

### Arrière plan flou <-> Arrière plan net (100 epochs)
<img src="imgs/output/blur/cgan.gif " width="200" height="200" />
<img src="imgs/output/blur/cganinv.gif " width="200" height="200" />

<hr>

### Style Ukiyo-e  <-> Photo (41 epochs)
<img src="imgs/output/Photo2style/dcgan.gif " width="200" height="200" />
<img src="imgs/output/Photo2style/dcganb.gif " width="200" height="200" />





## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Sources

* DCGAN : https://arxiv.org/pdf/1511.06434.pdf
* CycleGAN: https://arxiv.org/pdf/1703.10593.pdf
https://hardikbansal.github.io/CycleGANBlog/
* cGAN : https://arxiv.org/pdf/1611.07004.pdf
