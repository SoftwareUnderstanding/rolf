# SuperGlue-tasks-using-BERT
In this project we have implemented 2 SuperGlue tasks (RTE and BOOLQ) using BERT Model.


SuperGlue has total 10 Tasks. They are:
1) Broadcoverage Diagnostics (AX-b
2) CommitmentBank (CB)
3) Choice of Plausible Alternatives (COPA)
4) Multi-Sentence Reading Comprehension (MultiRC)
5) Recognizing Textual Entailment (RTE)
6) Words in Context (WiC)
7) The Winograd Schema Challenge (WSC)
8) BoolQ (BoolQ)
9) Reading Comprehension with Commonsense Reasoning (ReCoRD)
10) Winogender Schema Diagnostics (AX-g)

More information of those tasks can be found on https://super.gluebenchmark.com/tasks.

**Proposed Methodology:**

We are going to use the BERT model with the hugging face framework which is available in https://huggingface.co/transformers/model_doc/bert.html. BERT model is actually a multi-layer bidirectional Transformer encoder. The Transformer architecture is described in https://arxiv.org/pdf/1706.03762.pdf. It is an encoder-decoder network that uses self-attention on the encoder side and attention on the decoder side. The Transformer reads entire sequences of tokens at once while LSTMs read sequentially. BERT has 2 model sizes. One is BERT-BASE that contains 12 layers and another one is BERT-LARGE has 24 layers in the encoder stack. In the Transformer, the number of layers in an encoder is 6. After the encoder layer, both BERT model has Feedforward-network with 768 and 1024 hidden layer, respectively. Those two models have more self-attention heads (12 and 16 respectively) than Transformer. BERT-BASE contains 110M parameters while BERT-LARGE contains 340M parameters.

This model has 30,000 token vocabularies. It takes ([CLS]) token as input first, then it is followed by a sequence of words as input. Here ([CLS]) is a classification token. It then passes the input to the above layers. Each layer applies self-attention, passes the result through a feedforward network after then it hands off to the next encoder. The model outputs a vector of hidden size (768 for BERT-BASE and 1024 for BERT-LARGE ). If we want to output a classifier from this model, we can take the output corresponding to [CLS] token.

BERT is pre-trained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) on a large corpus from Wikipedia.  BERT was trained by masking 15% of the tokens with the goal to guess those words. For the pre-training corpus authors used the BooksCorpus which contains 800M words and English Wikipedia which contains 2,500M words.

BERT is helpful for different Natural Language Processing tasks like classification tasks, Name Entity Recognition, Part of Speech tagging, Question Answering, etc. But it is not useful for Language Models, Language Translation or Text Generation. BERT model is large and takes time for fine-tuning and inferencing.
