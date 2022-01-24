ROBERTA is a twist on Google's BERT model.

The following adjustments were made by the teams at Facebook AI and the University of Washington to help improve ROBERTA vs BERT.

1. Tokens were encoded with byte-level BPE and a vocabulary size of 50,000 sub-word units. There was no additional           preprocessing or tokenization of the inputs. 
2. The model was trained for longer and on more data with larger batch sizes than BERT.
3. Roberta was exposed to longer sequences of text than BERT during training.
4. Tokens were masked in a dynamic manner during training.
5. The loss function for the next sentence prediction was dropped. 
6. The Adam optimiser’s second moment was slightly lowered to help stabilise the model whilst training with larger batch sizes. Scientists also found training was very sensitive to  the magnitude of the Adam epsilon parameter.

In essence, both machine learning models are the best-in-class at reading and understanding language.  

In the accompanying file of this repo is a quick test of a ROBERTA pre-trained(MNLI) model.  

As background, I work with machine learning models to help analyse and classify electronic communications within the financial services industry domain. 

Aug/2019
   
