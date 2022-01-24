# Single Notebook Projects
In this repository, I shall upload the single notebook projects that I will learn in my learning career graph. This readme contains the learning outcome and useful resources for each project.

# [1] Fine-Tune-BERT-for-Quora Insincere Questions Classification-with-TensorFlow

<div align="center">
    <img width="512px" src='images/BERT_Layer.png' />
    <p style="text-align: center;color:gray">Figure 1: BERT Classification Model</p>
</div>

In this guided project I have learnt -
- How to efficiently handle data and create input pipelines with td.data for bert models.
- Tokenize and Preprocess Text for BERT using bert tokenizer.
- Fine-tune BERT for text classification with TensorFlow 2 and TF Hub.

Some useful resources to learn about bert and transformers:

1. The Illustrated Transformer (Jay Alammar)[https://jalammar.github.io/illustrated-transformer/]

2. The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)[https://jalammar.github.io/illustrated-bert/]

3. The Annotated Transformer (Harvard NLP)[https://nlp.seas.harvard.edu/2018/04/03/attention.html]

4. For more advanced learners, here's the original BERT paper: (BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)[https://arxiv.org/abs/1810.04805]


# [2] Create a Superhero Name Generator with TensorFlow

<div align="center">
    <img width="512px" src='../images/superhero.png' />
</div>


Text generation is a common natural language processing task. We will created a character level language model that will predict the next character for a given input sequence. In order to get a new predicted superhero name, we will need to give our model a seed input - this can be a single character or a sequence of characters, and the model will then generate the next character that it predicts should after the input sequence. This character is then added to the seed input to create a new input, which is then used again to generate the next character, and so on.


In this guided project I have learnt -
- How to create a character level language model that will predict the next character for a given input sequence.

<h3>I have also implemented a web based version of this project. See the flask deployment <a href="https://github.com/avishek-018/Superhero-Name-Generator">here</a>. </h3>
