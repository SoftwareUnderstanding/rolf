# Neural Machine Translation

**A full report of the task can be found in NMT_report.pdf**

This code implements a Sequence to Sequence Neural Machine Translation model based on [this paper](https://arxiv.org/pdf/1508.04025.pdf) by Luong et. al.
- The NMT model used global attention with dropout and input feeding.
- The model is also extended to implement the lexical model as per [this paper](https://arxiv.org/pdf/1710.01329.pdf)
- The data used for training is Japanese - English parallel corpus with Japanese as source language and English as target - language.
- The model will train until convergence and last epoch will be saved as checkpoint in the specified directory. The epoch with best validation loss will be saved as best checkpoint in the specified directory.
- Usage:
  - To train (with default settings):
   python train.py
  - To translate (with default settings):
   python translate.py
  - To calculate BLEU score:
   perl multi-bleu.perl -lc raw data/test.en < model translations.txt
   
   *test.en are the ground truth test data translations.

# References
- Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
- Toan Q Nguyen and David Chiang. Improving lexical choice in neural machine translation. arXiv preprint arXiv:1710.01329, 2017.


