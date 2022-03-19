# Date Translation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mike-a-yen/date-translation)

Sequence2Sequence model to translate date strings into the standard date format.
```
Dec 10, 1865 --> 1865-12-10
December 25, 1991 --> 1991-12-25
Saturday June 10 1741 --> 1741-06-10
```
This is for educational purposes to learn about seq2seq models and attention mechanisms.

### Helpful Resources
0. [Helpful primer with cool visuals](https://distill.pub/2016/augmented-rnns/)
1. [Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf)
2. [Luong et al.](https://arxiv.org/pdf/1508.04025.pdf)
3. [Conceptual implementation guide. ML Mastery](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
4. [Vaswani et al.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

A good Seq2Seq tutorial can be found [here](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

### Installation
1. Install <a href="https://docs.conda.io/en/latest/">conda</a>.
2. Create and activate conda env
```bash
conda create -n date python=3.7
conda activate date
```
3. Install python packages.
```
pip install -r requirements.txt
```

### Run
0. Activate the conda env.
```
conda activate date
```
1. Start your jupyter notebook.
```
jupyter notebook
```
2. Generate the dataset by running notebook `00_generate_data.ipynb`.
3. Complete the Seq2Seq w/ attention code in `01_FITB_seq2seq.ipynb`. A solution is found in `01_seq2seq.ipynb`.
4. Classify written dates w/ a transformer architecture. `02_transformers.ipynb`.

### Notes:
- Feel free to experiment with changing how the data is generated. You can widen or narrow the date range, or come up with different ways to represent dates as strings.
