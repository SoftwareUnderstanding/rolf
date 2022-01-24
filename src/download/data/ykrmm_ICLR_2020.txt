# Neural Phrase-To-Phrase Machine Translation implementation

Modèle de seq2seq learning basé sur des méchanismes d'attention.


## Sources intéressantes et état de l'art pour bien comprendre le modèle 


### Modèles de base sur Seq2seq et modèle d'attention
Seq2Seq Learning : Sutskever et al (2014) https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Attention based mechanism : Bahdhanau et al (2014) https://arxiv.org/pdf/1409.0473.pdf

### Architectures avancées qui utilisent des modèles d'attention
Papier sur les transformers : Attention is all you need https://arxiv.org/pdf/1706.03762.pdf


### Neural Phrase based-translation model 
Meilleur résultats sur les neural phrase based-translation model : Huang et al (2018) https://arxiv.org/pdf/1706.05565.pdf

### Segmental neural networks
Segmental neural networks : https://arxiv.org/pdf/1511.06018.pdf


## Experimentations

### Dataset 
IWSLT 2014 German to English dataset: 
https://wit3.fbk.eu/mt.php?release=2014-01 Aller à la ligne German et colonne English.

IWSLT 2015 English to Vietnamese Dataset : https://wit3.fbk.eu/mt.php?release=2015-01 Aller à la ligne english et colonne vietnamien. 

Une fois télécharger mettre dans un dossier iwslt2014_2015/

### Baseline (à réflechir si on implémente ou non):

NPMT : https://github.com/posenhuang/NPMT , Huang et al (2018) https://arxiv.org/pdf/1706.05565.pdf
Seq2Seq attention : https://github.com/pytorch/fairseq , Bahdhanau et al (2014) https://arxiv.org/pdf/1409.0473.pdf
Transformers : https://arxiv.org/pdf/1706.03762.pdf

### Metrique d'évaluations : 
BLEU. article https://www.aclweb.org/anthology/P02-1040.pdf
BLEU. Code pytorch https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/metrics/bleu.html

