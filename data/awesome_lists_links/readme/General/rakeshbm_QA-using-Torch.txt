# Query-based Information Extraction System using Deep Learning

This is an implementation of attention based End-To-End Memory Network architecture to build a Question Answering model.

The reference paper is MemN2N (https://arxiv.org/pdf/1503.08895v5.pdf - Sukhbaatar, 2015). The model is GPU-enabled and supports Adjacent Weight Tying, Position Encoding, Temporal Encoding and Linear Start.

The data corpus used is the collection of internal policies belonging to ITC Infotech Limited Organization. The goal was to provide employees an easier information access to the organization's policies.

The accuracy achieved was 93%. Here is a sample output:

![alt text](output.png)

## File Descriptions:

- preprocess.py: a Python script to index the data.
- train.lua: a Lua script to train and load the QA model.
- interact.lua: a Lua script to load the saved model and querying user requirements.
- model.dot: a grahical representation of the model configuration.
- output.png: a sample output screenshot.
