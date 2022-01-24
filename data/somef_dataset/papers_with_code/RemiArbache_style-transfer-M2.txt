# style-transfer-M2
## Projet académique d'expérimentation de Neural Style Transfer (Transfert de Style par réseau neuronal)

Basé sur [cet article](https://arxiv.org/abs/1705.06830) et [cette page TF Hub](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2). 

---
- Rémi ARBACHE 
- Paul BUCAMP
- Eugénie DALMAS

---

Lancement du serveur local (meilleures performances) :
```
python -m flask run
```

Version disponible [**en ligne**](https://ai-style-transfer.herokuapp.com/) hebergée par Heroku.

---
### Description
Ce projet a pour objectif la découverte du Neural Style Transfer (NST) à travers l’utilisation des modèles de d2l et de Google Brain. L'objectif est de réaliser un transfert d'une image de style à une image de contenu, et et de créer un site web d’application du modèle de Google Brain. 
Ce readMe permet d’expliquer la méthode de NST de d2l suivant un principe d’optimisation pour ensuite la comparer avec la méthode de Google Brain qui permet l’obtention en une boucle de l’image synthétisée sans entraînement préalable sur les images données en entrée.
Ce dépôt contient la mise en place d'un site web / API pour faire la démonstration de l’application du modèle choisi, créé par Google Brain, prenant des images en entrée pour donner en sortie l’image synthétisée.

### Introduction – Qu’est-ce que le Neural Style Transfer (NST) ?
Le NST permet, à partir d'une image de contenu (ex: une photo) et d'une image de référence de style (ex: une peinture), de créer une image qui maintienne le contenu de la première tout en reproduisant le style de la seconde. [Source](https://www.tensorflow.org/tutorials/generative/style_transfer )

Autrement dit, l’algorithme NST manipule des images dans le but de leur donner l’apparence ou le style visuel d’une autre image. Ces algorithmes utilisent des réseaux neuronaux profonds pour pouvoir réaliser la transformation d’images.

### Modèle d’optimisation avec d2l
Notre étude s'est tout d’abord portée sur le modèle NST de d2l se basant sur une optimisation par entraînement.

#### Données en entrée

Deux images (.png, .jpg)

#### Pré-traitements

Redimensionnement des images à la même taille : on garde le ratio de l'image de contenu en diminuant la taille si besoin afin de réduire la durée de l'entraînement.
Toute valeur de pixel inférieure à 0 ou supérieure à 1 est remplacée respectivement par 0 ou 1 pour standardiser chacun des trois canaux RVB d’une image donnée et transformer les résultats au format d’entrée du CNN.

#### Fonctionnement du modèle

De manière générale, le modèle tire parti des représentations en couches d’un réseau neuronal convolutif (CNN) afin d’appliquer automatiquement le style d’une image à une autre image. Pour cela, deux images sont utilisées : l’image de contenu et l’image de style. Un réseau neuronal est ensuite utilisé pour modifier l’image de contenu pour la rendre proche en style de l’image de style. Il s’agit donc d'optimiser l’image de contenu avec l’image de style.

La méthode présentée par d2l considère l’image synthétisée (en sortie) comme les paramètres d’un modèle, initialisé avec l’image de contenu. 

L’algorithme utilise un modèle pré-entraîné d’extraction hiérarchique de caractéristiques. Composé de plusieurs couches, on peut choisir la sortie de certains d’entre eux comme caractéristique de contenu ou de style. 

Schéma exemple avec un CNN d’extraction de caractéristiques à 3 couches :

![](images/CNN_example.png)

Les fonctions de perte sont calculées à travers une *forward propagation*, puis les paramètres du modèle (l’image synthétisée en sortie) sont mis jour à travers une *back propagation*. 

On compte 3 fonctions de perte : 

- *content loss* (rend l'image synthétisée et l'image de contenu proches en terme de contenu) ;
- *style loss* (rend l'image synthétisée et l'image de style proches en terme de style) ;
- *total variation loss* (aide à réduire le bruit dans l'image synthétisée).

A la fin de l’entraînement, les paramètres du modèle sont récupérés pour générer l’image synthétisée finale.

Pour des explications plus extensives sur le code Python du modèle, se référer au notebook disponible sur le repository résumant en français les explications de [d2l.ai](https://d2l.ai/chapter_computer-vision/neural-style.html). Ce notebook met en application la méthode décrite.

#### Résultats

| *Contenu*                                                    | *Style*                    | *Résultat*                          |
| ------------------------------------------------------------ | -------------------------- | ----------------------------------- |
| ![](images/rainier.jpg)                                      | ![style](images/style.jpg) | ![](images/synthesized_rainier.jpg) |
| <img src="images/paysage.jpg" alt="paysage" style="zoom:75%;" /> | ![](images/nuit.jpg)       | ![](images/synthesized_paysage.jpg) |

#### Conclusion et observations

La mise à jour itérative d'une image pour synthétiser une texture visuelle ou un transfert de style artistique à une image est une procédure d'optimisation lente. Elle exclut aussi toute possibilité d'apprentissage d'une représentation d'un style de peinture. De plus, la modification des (hyper)paramètres n'apportent pas beaucoup de changements sur l'image synthétisée à la fin de son optimisation.

### Modèle de prédiction avec Google Brain

#### Données en entrée
Deux images, tous formats.

#### Pré-traitements

Les images entrées par l'utilisateur sont d'abord réduites en taille pour réduire le temps de calcul (et le temps d'envoi pour la version en ligne). Un URI en base 64 est crée pour chaque image et envoyé au serveur avant d'être passé à la fonction de prédiction en elle-même. Enfin, unee fois chargées sous forme matricielle, la valeur de chaque pixel est normalisée.


#### . Fonctionnement du modèle

Les travaux sur le NST ont tout d’abord porté sur une méthode d'optimisation par mise à jour itérative de l’image synthétisée tel que pour la méthode de présentée ci-dessus.

Ensuite, d’autres travaux ont introduit un second réseau apprenant la transformation de l’image de contenu à sa version artistique. Ce réseau de transfert de style est un réseau de neurones convolutifs formulé dans la structure d'un codeur/décodeur. Le réseau résultant peut réaliser le transfert de style d’une image beaucoup plus rapidement, mais un réseau distinct doit être entraîné pour chaque style de peinture. Cela représente du gaspillage dans la mesure où certains styles de peintures partagent des textures, des palettes de couleurs ou sémantiques d’identification de scène communes. 

L’idée suivante a donc été de construire un réseau de transfert de style avec une architecture typique *encoder*/*decoder* mais spécialisant les paramètres de normalisation pour chaque style de peinture : *conditional instance normalization*, la normalisation de chaque unité d’activation, de sorte que la transformation linéaire de chaque peinture soit unique. Ainsi, un vecteur d’intégration (*embedding*) d’environ 300-d représente le style artistique d’une peinture.

En explorant cette question, un fait très surprenant a été trouvé sur le rôle de la normalisation dans les réseaux de transfert de style : pour modéliser un style, il suffit de spécialiser les paramètres de mise à l'échelle et de décalage après la normalisation à chaque style spécifique. En d'autres termes, tous les poids convolutifs d'un réseau de transfert de style peuvent être partagés entre de nombreux styles, et il suffit de régler les paramètres pour une transformation affine après normalisation pour chaque style.

Le modèle NST de Google Brain est donc la combinaison de modèles dont la méthode est d’apprendre la caractérisation de l’image de style aux paramètres de style directement. Dans le cas du modèle de Google Brain, pour faire le NST d’une image de style vers une image de contenu, deux réseaux sont utilisés : le *style transfer* <img src="https://render.githubusercontent.com/render/math?math=T(.%2C%5Cvec%20S)"> et le *style prediction* <img src="https://render.githubusercontent.com/render/math?math=P(.)">. 

![](images/NST_example.png)

**Réseau <img src="https://render.githubusercontent.com/render/math?math=T">:**  <img src="https://render.githubusercontent.com/render/math?math=T(.%2C%5Cvec%20S)"> peut ainsi faire le transfert de style en une seule propagation avant de n’importe quelle image de style à partir du moment qu’on connaît le vecteur <img src="https://render.githubusercontent.com/render/math?math=%5Cvec%20S%20%3D%20(%5Cgamma_s%2C%20%5Cbeta_s)">, calculé par le réseau <img src="https://render.githubusercontent.com/render/math?math=P(.)">.

<img src="https://render.githubusercontent.com/render/math?math=T(.%2C%5Cvec%20S)"> utilise la normalisation pour transformer une couche d'activation <img src="https://render.githubusercontent.com/render/math?math=z"> en une activation normalisée <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%20z"> spécifique à un style <img src="https://render.githubusercontent.com/render/math?math=s"> de sorte que <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%20z%20%3D%20%5Cgamma_s%5Cleft(%5Cfrac%7Bz%20-%20%5Cmu%7D%7B%5Csigma%7D%5Cright)%20%2B%20%5Cbeta_s">, où <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bz%20-%20%5Cmu%7D%7B%5Csigma%7D"> représente la couche <img src="https://render.githubusercontent.com/render/math?math=z"> normalisée, et <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%20z"> est la normalisation d'instance conditionnelle. Ils effectuent une mise à l'échelle et un déplacement à l'aide de vecteurs de paramètres dépendants du style.

**Réseau <img src="https://render.githubusercontent.com/render/math?math=P">:** Le réseau <img src="https://render.githubusercontent.com/render/math?math=P(.)"> fait la prédiction du vecteur <img src="https://render.githubusercontent.com/render/math?math=%5Cvec%20S"> à partir de l’image de style donnée en entrée. Il est basé sur le modèle pré-entraîné Inception-v3 : selon l’article, il calcule la moyenne à travers tous les channels d’activation de la couche Mixed-6e et retourne un feature vector sur lequel il applique 2 couches fully-connected pour prédire <img src="https://render.githubusercontent.com/render/math?math=%5Cvec%20S">.

#### Résultats

| *Contenu*                                                    | *Style*                    | *Résultat*                                 |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------ |
| ![](images/rainier.jpg)                                      | ![style](images/style.jpg) | ![](images/synthesized_rainier_google.jpg) |
| <img src="images/paysage.jpg" alt="paysage" style="zoom:75%;" /> | ![](images/nuit.jpg)       | ![](images/synthesized_paysage_google.jpg) |

#### Conclusion et observations

Cette méthode est très avantageuse, que ce soit par la qualité des résultats produits, par la vitesse de rendu, ou par l'étendue des styles compatibles. 

Des améliorations d'ordre pratique pourraient être faites au niveau de l'interface utilisateur, comme le choix de la taille de l'image de sortie.

### Site web d’application du modèle NST de Google Brain

Nous avons mis en place une plateforme d'essai du modèle de Google Brain en utilisant le code disponible sur TF Hub et en l'intégrant à un serveur Python en utilisant Flask. Le serveur dispose d'une interface web (HTML/CSS/JS) qui permet à l'utilisateur de choisir une image de style et une image de contenu et d'obtenir le résultat du transfert de style.

La version déployée en ligne peut présenter des performances réduites car Heroku ne supporte pas `tensorflow-gpu` et utilise donc `tensorflow-cpu`.

L'interface à vide se présente comme tel : 

![](images/preview_empty.png)

Cliquer sur les boutons *Browse* pour choisir une image de contenu et de style : 

![](images/preview_filled.png)

Une miniature de prévisualisation apparaît pour chaque image.



Cliquer sur le bouton *Predict* pour obtenir un résultat : 

![](images/preview_result.png)



###  Ouverture
Actuellement, les méthodes et modèles présentés réalisent le transfert de style d'une image à une autre image, typiquement d'une peinture ou texture à une photographie. Une autre application du NST serait l'application d'un style calligraphique à du texte. Ensuite, une possibilité serait l'application d'un style visuel (peinture/texture) en temps réel à une vidéo. 


---
### Sources
- Article Google Brain, *Exploring the structure of a real-time, arbitrary neural artistic stylization network*  : https://arxiv.org/pdf/1705.06830.pdf 
- Tutoriel de NST d2l : https://d2l.ai/chapter_computer-vision/neural-style.html 
- Inception-v3, *Rethinking the Inception Architecture for Computer Vision* : https://arxiv.org/pdf/1512.00567.pdf 
- Code Inception-v3 GitHub : https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L64 
- Ouverture, transfert de style à du texte, *Text Style Transfer: A Review and Experimental Evaluation* : https://arxiv.org/pdf/2010.12742.pdf

