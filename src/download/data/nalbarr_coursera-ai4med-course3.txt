# coursera-ai4med-course3
coursera-ai4med-course3

## TODOs
- Fix links
- Ensure large files are available for notebooks

## Week 1

### Key Concepts
- Analyze data from a randomized control trial
- Interpreting Multivariate Models
- Evaluating Treatment Effect Models
- Interpreting ML models for Treatment Effect Estimation
### Notebooks
- [week1a](week1/AI4M_C3_M1_lecture_nb_pandas.ipynb)
  - i.e., pandas refresher, slicing, read/update rows, read/update columns 
- [week1b](week1/AI4M_C3_M1_lecture_nb_sklearn.ipynb)
  - i.e., train/test split, simple model learn/fit/evaluate
  - i.e., dict args as **kwargs
  - i.e., itertools.product() to derive permutations
  - i.e., pass *args list
- [week1c](week1/AI4M_C3_M1_lecture_nb_logit.ipynb)
  - i.e., logistic regression

### Assignment
- [assignment1](week1/C3M1_Assigmment.ipynb)

### References
- [RCT](https://en.wikipedia.org/wiki/Randomized_controlled_trial)
- Levamisole and fluororacil background: https://www.nejm.org/doi/full/10.1056/NEJM199002083220602
- Data sourced from here: https://www.rdocumentation.org/packages/survival/versions/3.1-8/topics/colon
- C-statistic for benefit: https://www.ncbi.nlm.nih.gov/pubmed/29132832
- T-learner: https://arxiv.org/pdf/1706.03461.pdf

## Week 2
### Key Concepts
- Extracting disease labels from clinical reports
- Question Answering with BERT
### Notebooks
- [week2a](week2/AI4M_C3_M1_lecture_nb_logit.ipynb)

### References
- [BERT paper (Google, 2018)](https://arxiv.org/abs/1810.04805)
- [BERT Github](https://github.com/google-research/bert)
- [NegBio](https://github.com/ncbi-nlp/NegBio)
- [Grad cam])https://arxiv.org/pdf/1610.02391.pdf)
- Random forests + permutation importance: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf (R45f14345c000-1 Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.)
- Shapley importance: https://www.nature.com/articles/s42256-019-0138-9

## Week 3
### Key Concepts
- Interpreting Deep Learning Models
- Feature Importance in Machine Learning
### Notebooks
- [lesson3a]()
  - i.e., aggregate and group CCS codes; learn mappings

## Lesson 4
### Notebooks
- [lesson4a]()
  - i.e., transform line to encounter
  - i.e., pandas groupby()

### All References
- [Labeling methods and dataset](https://arxiv.org/abs/1901.07031)
- [Huggingface transformers library](https://github.com/huggingface/transformers)
- [BERT paper](https://arxiv.org/abs/1810.04805)
- [Question answering data set (used for example)](https://rajpurkar.github.io/SQuAD-explorer/)
- [Clinical note example for question answering](https://www.mtsamples.com/)
