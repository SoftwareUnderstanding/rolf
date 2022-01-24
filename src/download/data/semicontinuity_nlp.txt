# Training GPT-2 from scratch

Notebook trains GPT-2 from scratch in COLAB.

Also, as an experiment, GPT-2 model with tied weights in across transformed blocks is trained.
The intuition for that was, that weight tying might be benefitial, so that rules learned in one transformer block can be re-used in other blocks.

Here some results obtained:
*   Validation loss for vanilla GPT-2: 3.2658
*   Validation loss for GPT-2 with tied parameters in attention blocks: 3.5591

Experimental model has higher loss, but 6x less parameters in transformer block layers.

## Dataset and implementation details
* Dataset is wikitext-2
* GPT-2 model implementaion is taken from https://github.com/lopuhin/transformer-lm and adapted to Colab environment. Added code for weight tying experiment.

# References
* "Language Models are Unsupervised Multitask Learners", https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* "Attention is all you need", https://arxiv.org/pdf/1706.03762.pdf

## Steps to reproduce
Open the notebook lopuhin_transformer_lm/Training_GPT2_from_scratch.ipynb and follow it.
