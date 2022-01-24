# Extended XQuAD dataset for Multilingual Machine Comprehension Evaluation

## XQuAD
(Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance.
The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from
the development set of SQuAD v1.1 [(Rajpurkar et al., 2016)](https://www.aclweb.org/anthology/D16-1264/) together with their professional
translations into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi.
Consequently, the dataset is _entirely parallel_ across 11 languages.

For more information on how the dataset was created, refer to our paper,
[On the Cross-lingual Transferability of Monolingual Representations](https://arxiv.org/abs/1910.11856).

All files are in json format following the SQuAD dataset format. A parallel example in XQuAD in
English, Spanish, and Chinese can be seen in the image below. The full dataset consists of 240
such parallel instances in 11 languages.

## Extension: 
Multilingual Machine Comprehension (MMC) is a Question-Answering (QA) sub-task that involves quoting the answer for a question from a given snippet, where the question and the snippet can be in different languages. For evaluation in multilingual models (e.g. English and Hindi), you need to evaluate both mono-lingual performance (English Question-English Answer, same setup for Hindi), and cross-lingual performance (English Question-Hindi Answer and vice-versa). (Please note that this is different from XQuAD's definition of cross-lingual performance). The XQuAD dataset only contains monolingual variants, therefore, the script is written to generate cross-lingual variants from the monolingual evaluation sets. For our case https://arxiv.org/pdf/2006.01432.pdf, we create cross-lingual
variants using the monolingual variants of English and Hindi which are already present in XQuAD. 


## Data

This directory contains files in the following languages:
- Arabic: `xquad.ar.json`
- German: `xquad.de.json`
- Greek: `xquad.el.json`
- English: `xquad.en.json`
- Spanish: `xquad.es.json`
- Hindi: `xquad.hi.json`
- Russian: `xquad.ru.json`
- Thai: `xquad.th.json`
- Turkish: `xquad.tr.json`
- Vietnamese: `xquad.vi.json`
- Chinese: `xquad.zh.json`
- xquad_variant_generation.py: script to generate cross-lingual variants from the monolingual variants and adds to crosslingual_variants folder. Syntax 
```
python xquad_variant_generation.py -q <Question language code> -s <Snippet language code> 
```
Language codes are ar, de, en, etc. default: en

File generated for -q hi and -s en would be xquad.q_hi_s_en.json


## Reference

If you use this dataset, please cite 
[[1]](https://arxiv.org/pdf/2006.01432):

[1] Gupta, S. & Khade, N. (2020). [BERT Based Multilingual Machine Comprehension in English and Hindi](https://arxiv.org/pdf/2006.01432). arXiv preprint arXiv:2006.01432.

```
@misc{gupta2020bert,
    title={BERT Based Multilingual Machine Comprehension in English and Hindi},
    author={Somil Gupta and Nilesh Khade},
    year={2020},
    eprint={2006.01432},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

And 

[[2]](https://arxiv.org/abs/1910.11856):

[2] Artetxe, M., Ruder, S., & Yogatama, D. (2019). [On the cross-lingual transferability of monolingual representations](https://arxiv.org/abs/1910.11856). arXiv preprint arXiv:1910.11856.

```
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
```

## License

This dataset is distributed under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

This is not an officially supported Google product.
