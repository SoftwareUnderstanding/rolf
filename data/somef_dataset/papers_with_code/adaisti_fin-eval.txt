### Evaluation datasets for Finnish word vectors

This repository contains 3 evaluation datasets that can be used for evaluating Finnish word vectors on semantic similarity and analogy tasks. The datasets are translated from original English sources. They were created for my Master's Thesis, "Alimerkkijonot suomen sanavektorien tuottamisessa neuroverkoilla" (2018).

### Semantic similarity

The folder contains the datasets WS353 and WS277. WS353 is a complete Finnish translation of the WS353 dataset originally published by Finkelstein et al (https://dl.acm.org/citation.cfm?id=503110). WS277 is a shortened version of WS353 where word pairs with non-obvious translations are removed.

The dataset is formatted as follows: English word1, English word2, average semantic similarity by human evaluators on scale 0 - 10, Finnish word 1, Finnish word2. On WS353, rows that were removed in WS277 also contain a "!" mark. 

### Analogies

The dataset SSWR-fi is a shortened translation of the Semantic-Syntactic Word Relations dataset originally published by Mikolov et al (https://arxiv.org/abs/1301.3781). Analogy tasks and word pairs with non-obvious Finnish translations were removed in the translation process.

The dataset contains analogy type sections that are titled with :analogy-type. After that, on every row there are two word pairs that have a similar analogy relation, such as "man woman king queen", where the analogy type is male-female. The dataset is in Finnish only.
