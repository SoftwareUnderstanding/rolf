# Fully Attention-Based Information Retriever (FABIR)
FABIR is a reading comprehension model introduced in this [IJCNN 2018 paper](https://arxiv.org/abs/1810.09580). It was designed for the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset with the goal of achieving low training and inference times, being at least 5 times faster than competing models. FABIR was inspired by [Google's transformer](https://arxiv.org/abs/1706.03762) and includes no recurrence. It is a simple feedforward network made powerful by attention mechanisms.

## Requirements
- gloves: https://nlp.stanford.edu/projects/glove/ - Currently using only the 6B corpus
- json
- numpy (punkt)
- nltk
- pandas
- tensorflow 1.5 (FABIR does not support later versions yet) 
- tqdm

## Citation
If you find FABIR useful please cite us in your work:

    @inproceedings{Correia2018,
      author = {Correia, Alvaro H. C. and Silva, Jorge L. M. and De Martins, C. Thiago and Cozman, G. Fabio},
      booktitle = {Proceedings of the International Joint Conference on Neural Networks},
      pages = {2799--2806},
      publisher = {IEEE},
      title = {A Fully Attention-Based Information Retriever},
      year = {2018}
    }
