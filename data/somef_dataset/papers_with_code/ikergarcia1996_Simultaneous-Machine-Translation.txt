# Simultaneous-Machine-Translation
## The project
This is a project made by Iker García for the Advanced applications on language technologies course. This course is taught in the master of natural language processing in the EHU/UPV.

## The idea
In the educational world there exists a big problem, the problem of different languages. For example, our university offers courses in Basque to those who do not know the language can not attend. There are also courses in Spanish that people who come from other countries (Erasmus) can not attend. And these are just some examples, the language barrier is a big problem in the educational world. That’s why my project focuses in trying to solve this problem. My idea is to implement a simultaneous translator that can automatically translate on the go a conversation from one language to another. The application will be able to listen to someone speaking and it will display as text what he has said translated to another language

## How the simultaneous translation works
The simultaneous translation is composed of two modules, the “speech to text” module and the “translation” module.

Speech to text: For the “speech to text” module I will use the Speech recognition API provided by Google (https://cloud.google.com/speech-to-text/)

Translator: The translator is based in the Transformer model (Vaswani et al 2017. "Attention Is all you need" https://arxiv.org/abs/1706.03762). I have replicated the model using the Pytorch API.

Parallel data: To train the model I have used the OpenSubtitles v2018 corpus available here: http://opus.nlpl.eu/.This corpus contains 64,7M sentences aligned for Spanish and English.

Resources used for the implementation
"Attention Is all you need": https://arxiv.org/abs/1706.03762
How to code The Transformer in Pytorch by Samuel Lynn-Evans: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
How to use TorchText for neural machine translation, plus hack to make it 5x faster by Samuel Lynn-Evans: https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95
The transformer - Attention is all you need by Michał Chromiak: https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XFW52rpKhhE

The Illustrated Transformer by Jay Alammar: http://jalammar.github.io/illustrated-transformer/

The Annotated Transformer by Alexander Rush: http://nlp.seas.harvard.edu/2018/04/03/attention.html

