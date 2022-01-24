## Generative GPT Model

#### Papers

*Attention Is All You Need*, Vaswani et Al., Google Brain, 2017. 
- Link: https://arxiv.org/abs/1706.03762

*Improving Language Understanding by Generative Pre-Training*, Radford et Al., OpenAI, 2018
- Link https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

*Language Models are Unsupervised Multitask Learners*, Radford et Al., OpenAI, 2019
- Link: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf


Original GPT model
- 12 transformer layers
- Embedding dimension of 768
- 12 attention heads
- Feed forward dimension of 3072
- Adam optimizer with max learning rate of 2.5e-4
- Trained for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens
- Gelu activation function

Attention is All You Need model
- Vocab of about 37000 tokens
- Base model trained for 100,000 steps, with 4000 warm up steps
- Quality drops off with too many attention heads