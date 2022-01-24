# WordEmbedding

## There are two main objectives of this project:

1. Compare existing Word Embedding methods (including Word2Vec, Glove, FastText and ELMo) on Chinese datasets. It starts by finding usable Chinese textual data, then find python platforms that easily allow the processing of natural language in Chinese (NLTK, Fast AI, HanLP, jieba, SnowNLP, gensim). It will then be necessary to propose good criteria for comparing the algorithms and to make the algorithms work on the datasets. It will also be necessary to identify existing models in the final report.
2. Try to use Transformer (https://arxiv.org/abs/1706.03762) instead of LSTM to train a bi-direnctional language model. It will then be necessary to compare the proposed approach with the original ELMo on 6 different tasks used to evaluate it (https://allennlp.org/elmo). This idea will be tested first in English and then in Chinese, if the results allow.

## Les objectifs du PAO sont doubles : 

1. Comparer les  méthodes de Word Embedding (« plongement de mots » ou « plongement lexical » en français d'après wikipédia)  existantes (notamment Word2Vec, Glove, FastText et ELMo) sur des jeux de données en chinois. Il faudra commencer par trouver des données textuelles en chinois utilisables, puis trouver les plateformes python qui permettent facilement le traitement de la langue naturelle en chinois (NLTK, Fast AI, HanLP, jieba, SnowNLP, gensim). Il faudra ensuite proposer des bons critères de comparaison des algorithmes et faire fonctionner les algorithmes sur les données. Il faudra aussi identifier les modèles existants dans la littérature. 
2. Adapter Transformer (https://arxiv.org/abs/1706.03762), un des modèles utilisé pour la traduction automatique. L'idée est d'essayer d'utiliser le modèle Transformer à la place des classiques réseaux de type « long short term memory (LSTM) »  pour entraîner un modèle de language bi-directionnel. Il faudra ensuite comparer l'approche proposée avec l'ELMo original sur 6 differentes tâches utilisée pour l'évaluer (https://allennlp.org/elmo). Cette idée sera d'abord testée en anglais puis en chinois, si les résultats le permettent.
