# Introduction
We tackle the problem of OCR post processing. In OCR, we map the image form of the document into the text domain. This is done first using an CNN+LSTM+CTC model, in our case based on tesseract. Since this output maps only image to text, we need something on top to validate and correct language semantics.

The idea is to build a language model, that takes the OCRed text and corrects it based on language knowledge. The langauge model could be:
- Char level: the aim is to capture the word morphology. In which case it's like a spelling correction system.
- Word level: the aim is to capture the sentence semnatics. But such systems suffer from the OOV problem.
- Fusion: to capture semantics and morphology language rules. The output has to be at char level, to avoid the OOV. However, the input can be char, word or both.

# Fusion model
The fusion model target is to learn:

     p(char | char_context, word_context)
 
We first use seq2seq vanilla Keras implementation, adapted from the lstm_seq2seq example on Eng-Fra translation task. The adaptation involves:

- Adapt to spelling correction, on char level
- Pre-train on a noisy, medical sentences
- Fine tune a residual, to correct the mistakes of tesseract 
- Limit the input and output sequence lengths
- Enusre teacher forcing auto regressive model in the decoder
- Limit the padding per batch (TODO)
- Learning rate schedule (TODO)

# Usage and description
Spell corrector.ipynb

# Results
## Sample results:

|OCR sentence|GT sentence|Decoded sentence|
---------------|-----------|----------------|
|Therapy: 21aJn2018k to Recorded|Therapy: 21Jan2018 to Recorded|Therapy: 21Jan2018 to Recorded|
|Diagpnosi:s|Diagnosis:|Diagnosis: ICD Code:|
|Hospital Service Squqrgery|Hospital Service Surgery|Hospital Service Surgery|
|Currentf Meds|Current Meds|Current Meds|
|yACCIDENT DETAIL|ACCIDENT DETAILS|ACCIDENT CLAIM FORM|
|aPst Medizcalisetory|Past Medical History|Past Medical History|

The results is on unseen test data, which included tesseract output + corrections, in addition to noisy synthetic data from the GT.

Main modifications:

•	Fine tune a residual, to correct the mistakes of tesseract, instead of training on generic data (books like in big.txt)

•	Limit the input and output sequence lengths (4:40 chars)

# Next steps
- Add attention
- Full attention
- Condition the Encoder on word embeddings of the context (Bi-directional LSTM)
- Condition the Decoder on word embeddings of the context (Bi-directional LSTM) 

# References
- Sequence to Sequence Learning with Neural Networks
     https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
     RNN Encoder-Decoder for Statistical Machine Translation
     https://arxiv.org/abs/1406.107
