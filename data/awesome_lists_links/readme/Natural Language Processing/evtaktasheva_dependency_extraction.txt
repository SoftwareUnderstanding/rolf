## Shaking Syntactic Trees on the Sesame Street: Multilingual Probing with Controllable Perturbations

The [paper](http://arxiv.org/abs/2109.14017) is accepted to the 1st Workshop on Multilingual Representation Learning ([MRL](https://www.sites.google.com/view/mrl-2021)) at EMNLP 2021. 


### Tasks
The paper  proposes nine  probing  datasets  organized  by  the  type of controllable text perturbation for three Indo-European languages with a varying degree of word order flexibility:  nglish (West Germanic, analytic), Swedish (North Germanic, analytic), and Russian (Balto-Slavic, fusional).

1. The (**NShift**) task tests the LM sensitivity to *local* perturbations taking into account the syntactic structure.
2. The (**ClauseShift**) task probes the LM sensitivity to *distant* perturbations at the level of syntactic clauses. 
3. The (**RandomShift**) task tests the LM sensitivity to *global* perturbations obtained by shuffling the word order.

### Models
The experiments are run on two 12-layer multilingual Transformer models released by the HuggingFace library:

1. **M-BERT** [(Devlin et al. 2019)](https://arxiv.org/abs/1810.04805), a transformer model of the encoder architecture, trained on multilingual Wikipedia data using the Masked LM (MLM) and Next Sentence Prediction pre-training objectives.
2. **M-BART** [(Liu et al. 2020)](https://arxiv.org/abs/2001.08210), a sequence-to-sequence model that comprises a BERT encoder and an autoregressive GPT-2 decoder \cite{radford2019language}. The model is pre-trained on the CC25 corpus in 25 languages using text infilling and sentence shuffling objectives, where it learns to predict masked word spans and reconstruct the permuted input. We use only the encoder in our experiments.

### Experiments

1. **Parameter-free Probing**: We apply two unsupervised probing methods to reconstruct syntactic trees from self-attention (**Self-Attention Probing**) [(Htut et al., 2019)]()) and so-called "impact" (**Token Perturbed Masking**) [(Wu et al., 2020)](https://arxiv.org/pdf/2004.14786https://arxiv.org/abs/1911.12246?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529.pdf)) matrices computed by feeding the MLM models with each sentence `s` and its perturbed version `s'`.
2. **Representation Analysis**: We use two of the metrics proposed by [(Hessel and Schofield, 2021)](https://aclanthology.org/2021.acl-short.27/) to compare contextualized representations and self-attention matrices produced by the model for each pair of sentences `s` and `s'`. **Token Identifiability** (TI) evaluates the similarity of the LM's contextualized representations of a particular token in `s` and `s'`. **Self-Attention Distance** (SAD) measures if each token in `s` relates to similar words in `s'` by computing row-wise Jensen-Shannon Divergence between the two self-attention matrices.
3. **Pseudo-perplexity**: Pseudo-perplexity (PPPL) is an intrinsic measure that estimates the probability of a sentence with an MLM similar to that of conventional LMs. We use two PPPL-based measures under [implementation](https://github.com/jhlau/acceptability-prediction-in-context) by  [(Lau et al. 2020)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00315/96455/How-Furiously-Can-Colorless-Green-Ideas-Sleep) to infer probabilities of the sentences and their perturbed counterparts.
 
### Positional Encoding
 
We aim at analyzing the impact of the PEs on the syntactic probe performance. Towards this end, we consider the following three configurations of PEs of the M-BERT and M-BART models: (1) **absolute**=frozen PEs; (2) **random**=randomly initialized PEs; and (3) **zero**=zeroed PEs.

 ### Results

1. **The syntactic sensitivity depends upon language** At present, English remains the focal point of prior research in the field of NLP, leaving other languages understudied. Our probing experiments on the less explored languages with different word order flexibility show that M-BERT and M-BART behave slightly differently in Swedish and Russian.
2. **Pre-training objectives can help to improve syntactic robustness** Analysis of the M-BERT and M-BART LMs that differ in the pre-training objectives shows that M-BERT achieves higher δ UUAS performance across all languages as opposed to M-BART pre-trained with the sentence shuffling objective.
3. **The LMs are less sensitive to more granular perturbations** The results of the parameter-free probing show that M-BERT and M-BART exhibit little to no sensitivity to *local* perturbations within syntactic groups (**NgramShift**) and *distant* perturbations at the level of syntactic clauses (**ClauseShift**). In contrast, the *global* perturbations (**RandomShift**) are best distinguished by the encoders. As the granularity of the syntactic corruption increases, we observe a worse probing performance under all considered interpretation methods.
4. **M-BERT and M-BART barely use positional information to induce syntactic trees** Our results under different PEs configurations reveal that M-BERT and M-BART do not need the precise position information to restore the syntactic tree from their internal representations. The overall behavior is that zeroed (except for M-BERT) or even randomly initialized PEs can result in the probing performance and one with absolute positions.


## Setup and Usage

TBA

## Cite us

```
@inproceedings{taktasheva-etal-2021-shaking,
    title = "Shaking Syntactic Trees on the Sesame Street: Multilingual Probing with Controllable Perturbations",
    author = "Taktasheva, Ekaterina  and
      Mikhailov, Vladislav  and
      Artemova, Ekaterina",
    booktitle = "Proceedings of the 1st Workshop on Multilingual Representation Learning",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.mrl-1.17",
    pages = "191--210",
    abstract = "Recent research has adopted a new experimental field centered around the concept of text perturbations which has revealed that shuffled word order has little to no impact on the downstream performance of Transformer-based language models across many NLP tasks. These findings contradict the common understanding of how the models encode hierarchical and structural information and even question if the word order is modeled with position embeddings. To this end, this paper proposes nine probing datasets organized by the type of controllable text perturbation for three Indo-European languages with a varying degree of word order flexibility: English, Swedish and Russian. Based on the probing analysis of the M-BERT and M-BART models, we report that the syntactic sensitivity depends on the language and model pre-training objectives. We also find that the sensitivity grows across layers together with the increase of the perturbation granularity. Last but not least, we show that the models barely use the positional information to induce syntactic trees from their intermediate self-attention and contextualized representations.",
}
```
