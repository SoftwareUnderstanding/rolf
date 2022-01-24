# Chatbot-Reformer
---

Building a chatbot using the Reformer model by Trax

[![conversation-demo.png](https://i.postimg.cc/SQPsXGSZ/conversation-demo.png)](https://postimg.cc/grvdQ86R)
<p align="center">
    An actual response from my Reformer chatbot
</p>


## Project Summary

1. This project built a chatbot by using the Reformer architecture and trained on the MultiWOZ dataset from scratch.

2. After training on the MultiWOZ dataset for 20,000 epochs, the model achieved the **accuracy of 67.15%** on the test set (training/test size = 9938/500).

3. The chatbot learned to have a conversation with the custumer for helps on purchasing train tickets, booking a restaurant, giving directions, etc.

4. By feeding in some custom stencences, the **chatbot is able to handle the conversation and address the request**.

## Data

I used the [MultiWOZ: Multi-Domain Wizard-of-Oz dataset (ver 2.1) dataset](). The dataset contains dialogues such as the following:

```
Hello, I have been robbed.  Can you please help me get in touch with the police?
Parkside Police Station is in Parkside, Cambridge. Their number is 01223358966. Anything else I can do for you?
Can I please have the postcode as well?
The postcode for the Parkside Police Station is CB11JG. Can I help you with anything else?
Was Parkside the address of the police station? If not, can I have the address please?
Yes, Parkside is the address.
Thank you that will be all for now.
Great. Thank you for contacting Cambridge Towninfo Centre.
You were great. Goodbye.
We are happy to help. Have a good day!
```

## Data Preprocessing

The raw dialogues were preprocessed by adding `Person 1: ` and `Person 2:` into the beginning of each sentence.

The data pipeline consists of:

1. shuffling
2. tokenization
3. length filtering
4. bucketing


## Model

I used a [Reformer Language Model](https://trax-ml.readthedocs.io/en/latest/trax.models.html#trax.models.reformer.reformer.ReformerLM) for the task. This model only uses the decoder.

```python

def ReformerLM(vocab_size = 33000, n_layers = 6, mode = 'train', attention_type = tl.SelfAttention):

    """
    Create the Reformer

    Inputs
            vocab_size: <int> the size of the vocabulary
            n_layers: <int> number of decoder layers
            mode: <str> 'train', 'eval', or 'predict'
            attention_type: <trax class> type of the attention class

    Output
            model: <trax model> the Reformer model
    """

    model = trax.models.reformer.ReformerLM(vocab_size = vocab_size,
                                            n_layers = n_layers,
                                            mode = mode,
                                            attention_type = attention_type)

    return model

## Check
model = ReformerLM(mode = 'train')
```

## Results

I trained the model from scrtch for 20,000 steps. The model achieved 67.15% accuracy.

Below shows the model's output from the test sentences:

### Dialogue #1
```
(Input)
Person 1: Where is the Japanese restaurant?

(Output)
Person 2: The Addenbell is located at 32 Bridge Street City Centre. Would you like me to book a table?
Person 1: No, I'd like to book a table for 7 people at 14:00 on Saturday.
Person 2: I have booked you a table for 7 at the Bedouin. Your reference number is XYZ1Z.
Person 1: Thanks, that's all I need.
Person 2: You're welcome
```

### Dialogue #2
```
(Input)
Person 1: Where can I find the post office?

(Output)
Person 2: The Cambridge Belfry is located at 5 Greens Road.
Person 1: What is the postcode?
Person 2: The postcode is cb21sj.
Person 1: Thanks, that's all I need.
Person 2: You're welcome. Have a great day!and enjoy your day!us.
Person 1: Thanks, you too.
Person 2: You're welcome. Have a great day!.Person 1
```

### Dialogue #3
```
(Input)
Person 1: Person 1: Hello, I'm wondering if I can book a hotel room tonight?

(Output)
Person 2: I can help with that. What area of town would you like to stay in?
Person 1: I'd like to stay in the east, please.
Person 2: I have three options for you. The Huntingdon Marriott Hotel, and the rest of your stay.
Person 1: I'd like to book it for 2 people and 4 nights starting from wednesday.
Person 2: I have booked your hotel for 4 people for 2 nights starting on Saturday. Your reference number is X1Z1Z1.
Person 1: Thanks, that's all I need.
Person 2: You're welcome. Have a great day!!!!!!.
Person 1: Thank you, goodbye.
Person 2: Thank you for using our service. Have a great day!.
Person 1: Thanks, you too.
Person 2: You're welcome.
```

## References

This notebook was learned and modified from the assignment of the course [Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp) on *Coursera* with the following amendments:

1. Instead of using a pre-trained model, I trained the chatbot **from scratch**

2. Instead of using `trax.supervised.decoding.autoregressive_sample_stream`, I used a helper function to continue the conversation token by token.

3. I cleaned up and rewrited the part of model evaluation. The model can be used to predict on the test dataset, or any custom starter sentence.

**The Reformer paper**

N. Kitaev et al., **Reformer: The Efficient Transformer**: https://arxiv.org/abs/2001.04451



---
Last updated: 11/03/2021
