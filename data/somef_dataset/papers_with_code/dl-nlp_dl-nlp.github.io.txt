- Zeit: Mi. 9-11 (VL), Mi. 11-12 (Ü)
- Raum: 131
- Start: 16.10.2019

Sie erreichen Dozent und Tutor unter profilmodul1920 (at) cis (dot) lmu (dot) de

## Topics

The topics of this lecture are the foundations of deep learning, with a particular focus on practical aspects and applications to natural language processing and knowledge representation.

- Foundations of machine learning:
  - Loss functions
  - linear regression
  - logistic regression
  - gradient-based optimization
  - neural networks and backpropagation
- Deep learning tools:
  - Numpy, PyTorch, Keras
- Deep Learning Architectures for NLP:
  - RNNs, CNNs, Self-Attention (Transformer)
- Applications:
  - Tagging, Sentiment-Prediction, Relation extraction
- Practical projects (NLP related, to be agreed on during the course)

## Klausur
 - Die Klausur findet am Mittwoch 8.1.2020 um 10:00 statt und dauert 90 Minuten. **Raum: Theresienstr. 41 C 112**
 - Die Nachklausur findet am Samstag 22.2.2020 um 10:00 statt und dauert 90 Minuten. **Raum: Hörsaal BU101, Oettingenstr. 67**

## Nachklausur-Ergebnisse
- Ergebnisse der Nachklausur [klausurergebnisse](ergebnisse_nachklausur.md)

### Bewertung
 - Bewertung:
   - Zum Bestehen sind mindestens 50% der möglichen Klausurpunkte nötig (ohne Bonuspunkte!). Eine 1.0 ist auch ohne Bonuspunkte möglich. 
   - Bei bestandener Klausur werden nach der folgenden Formel Bonuspunkte addiert:
   - klausurpunkte_mit_bonus = g_exam + max(0, 2 * (g_bonus - 0.5)) * M / 10
   - g_bonus = 0.67 * g_exercises + 0.33 * g_project
 - Erklärung:
   - Es werden nur Bonuspunkte oberhalb von 50% der möglichen Bonuspunkte angerechnet.
   - Die angerechneten Bonuspunkte zählen für bis zu 10% der Klausurpunkte.
   - g_exercises: Anteil der erreichten Punkte in den Übungen
   - g_project: Anteil der erreichten Punkte im Projekt
   - g_exam = Erreichte Punkte Klausur
   - M = Mögliche Punkte Klausur
   



## Course Material

| Date | slides | homework | materials |
|-----------------------------|:--------------------------------:|:------:|:-------------------------------------------------------------------|
| Oct. 16, 2019 | Machine learning overview [pdf](ml_basics_I.pdf)| [Linear Algebra](ex01_linalg.pdf) | |
| Oct. 23, 2019 | Machine learning overview II [pdf](ml_basics_II_short.pdf) | [Probability Theory](ex02_probability.pdf) |
| Oct. 30, 2019 | Machine learning overview III [pdf](ml_basics_III.pdf); Numpy [pdf](numpy_intro.pdf) | [Numpy](numpy.ipynb) - Abgabe bitte zu zweit oder zu dritt |  |
| Nov. 06, 2019 | Machine learning overview III [pdf](ml_basics_III_short.pdf); CIP guide [pdf](guide_cip.pdf) | [Pytorch](pytorch_intro.ipynb) - Abgabe bitte zu zweit oder zu dritt **bis 19.11. [Correction of Typo in Ex.1: normalize the data so that each feature has 0 mean and unit standard deviation]** | Merging of Variable and Tensor [http](https://pytorch.org/blog/pytorch-0_4_0-migration-guide/) |
| Nov. 13, 2019 | Backpropagation [pdf](ml_basics_backpropagation.pdf); PyTorch into [pdf](pytorch_intro.pdf) |  |
| Nov. 20, 2019 | Word2Vec [pdf](05_word2vec_corrected.pdf) -- correction: removed misleading indices on slide 25; Keras [pdf](07_keras.pdf) | [Word2Vec](pytorch_wordEmbeddings.ipynb) - Abgabe bitte zu zweit oder zu dritt |
| Nov. 27, 2019 | RNN Basics [pdf](rnn_handout.pdf); CNN [pdf](cnn_handout.pdf) | [pdf](exercises_ex06_lstm_ex06_lstm.pdf) |
| Dez. 04, 2019 | Keras (slides: Nov. 20, 2019) | [Keras tagging](argument_tagging.ipynb) -- Abgabe bitte zu zweit oder zu dritt -- [atis](atis.json) |
| Dez. 11, 2019 | Attention [pdf](08_attention.handout.pdf) | | [Test Exam / Probeklausur](probe_klausur.pdf) -- There will be a Q&A Session next week. |
| Dez. 18, 2019 | | | Relation extraction exercise [pdf](relation_project.pdf); [relation_project.zip](https://cis.uni-muenchen.de/~beroth/relation_project.zip) |
| Jan. 15, 2020 | Project Q&A [notes](notes_20200115.pdf)| | [count_params.py](count_params.py) |
| Jan. 29, 2020 | Help with projects | | |
| Feb. 5, 2020 | Project presentations | | |


## More materials
- Jupyter Notebooks Tips and Tricks: [http](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- Essence of Linear Algebra: [youtube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Keras functional API: [http](https://keras.io/getting-started/functional-api-guide/)
- Bengio: Practical recommendations ... [arXiv](https://arxiv.org/abs/1206.5533)
- Socher: Neural Tips and Tricks [pdf](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)
- Regularisierung: [reg](reg.md)
