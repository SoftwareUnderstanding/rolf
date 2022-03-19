# Apprentissage profond par renforcement

## M2 IA Université lyon 1

Ce dépot contient une implémentation de l'algorithme DQN tel qu'il est décrit dans le papier de recherche
[Mnih et al., 2015]. Cet algorithme ne converge pas toujours et peut
osciller voir diverger notament à cause de la forte correlation
etat/action.  
Pour contourner ce problème nous avons utilisé deux
techniques inspirées du travail cité ci dessus:
* Expérience replay (Un buffer d'expériences dans lequel on tire aleatoirement les données à utiliser pour l'appentissage).
* Fixed Q-Target (Un deuxième réseau de neuronne dérivé du premier pour
calculer Q̂(s',a'))

L'algorithme à été testé sur deux environnements Gym (https://gym.openai.com/):
* CartPole-v1 (avec un fully connected Q-Network classique)
* BreakoutNoFrameskip-v4 (à partir de l'image avec un Q-Network convolutionel)

## Contenu du dépot

* logs/ : contient les traces videos et autres fichiers de logs issus de l'entrainement des modèles.
* saved_params/ : contient les poids sauvergardés des modèles après entrainement pour une utilisation ultérieure
* breakout_agent.py | cartPole_agent.py : contient le code des agents pour les deux envirronements
* dql.py fichier à lander pour entrainer les modèles
* test_from_file.py fichier à lancer pour tester les modèles à partir des poids déja enregistrés dans saved_param/
* qNetwork.py | convolutionalQNetwork.py : contient les modèles des réseaux profond (fully connected et convolutionnel)
* replay_memory : Contient le buffer utilisé pour l'expérience replay
* wrapper.py : Contient les wrappers qui permettent de traiter l'environnement BreakoutNoFrameskip-v4 avec des paramètres
inspirés du papier de recherche [Mnih et al., 2015].
* requirements.txt : Contient les dépendances à installer pour faire fonctionner le code dans un environnement virtuel.

## Instructions d'utilisation

Pour tester le modèle pré-entrainé sur les environnements il faut lancer le fichier test_from_file.py:
* Test sur CartPole-v1:
```console
python test_from_file.py CartPole-v1
```
* Test sur BreakoutNoFrameskip-v4
```console
python test_from_file.py
```

Pour entrainer le modèle, il suffit de lancer le fichier dql.py:
* Entrainement sur CartPole-v1:
```console
python dql.py CartPole-v1
```
* Entrainement sur BreakoutNoFrameskip-v4
```console
python dql.py
```

## Dépendances

Pour installer les dépendances:
```console
pip install -r /path/to/requirements.txt
```

## Sources
[Mnih et al., 2015] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-
level control through deep reinforcement learning. Nature, 518(7540) :529.

[Wang et al., 2015] Wang, Z., Schaul, T., Hessel, M., Van Hasselt, H., Lanctot, M., and
De Freitas, N. (2015). Dueling network architectures for deep reinforcement learning.
arXiv preprint arXiv :1511.06581. (voir: https://arxiv.org/abs/1511.06581)