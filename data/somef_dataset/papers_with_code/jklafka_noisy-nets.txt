# A better sentence-level autocorrect: sequential denoising autoencoders with language models
In this project, we use state-of-the-art linguistic encoders such as BERT and GPT-2 to denoise corrupted text within a sequential denoising autoencoder architecture and training objective. What does that mean? For example, in this sentence:

"Where is the the teacher?"

you can easily tell that the intended *message* is "where is the teacher". You unconsciously deletes the repeated "the" from the sentence with minimal, if any, mental effort.

To approach this task, I train an *autoencoder*: a network that recreates the sentence I give it as input. Since I feed the autoencoder words one by one, it is a *sequential autoencoder*. Finally, I give the autoencoder noisy input i.e. corrupted text, where a word has been deleted or inserted, or two words have been swapped, and ask it to give me the original, uncorrupted text back. This means that the autoencoder *denoises*.

How do powerful language models such as BERT encode noisy sentences? Can these language models help us denoise sentences at scale? What does this process reveal about the model's knowledge of language in general?

## How this repository is organized

*preprocess.sh*: This script introduces noise into a corpus you provide and creates a vocabulary file. Afterwards, you can run the autoencoder (either train your own or use a provided model) on the preprocessed corpus. s
