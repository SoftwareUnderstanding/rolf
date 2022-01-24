# IMPLEMENTATION OF BLACK BOX ATTACK WITH TRANSFERABLE MODEL BASED EMBEDDING
Zhichao Huang, Tong Zhang
The Hong Kong University of Science and Technology

https://openreview.net/forum?id=SJxhNTNYwB

Abstract : "We present a new method for black-box adversarial attack. Unlike previous methods that combined transfer-based and scored-based methods by using the gradient or initialization of a surrogate white-box model, this new method tries to learn a low-dimensional embedding using a pretrained model, and then performs efficient search within the embedding space to attack an unknown target network. The method produces adversarial perturbations with high level semantic patterns that are easily transferable. We show that this approach can greatly improve the query efficiency of black-box adversarial attack across different target network architectures. We evaluate our approach on MNIST, ImageNet and Google Cloud Vision API, resulting in a significant reduction on the number of queries. We also attack adversarially defended networks on CIFAR10 and ImageNet, where our method not only reduces the number of queries, but also improves the attack success rate."


## STATE OF THE ART 


### White-box attack

### Black-box attack


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

