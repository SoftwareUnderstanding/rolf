# SelfAttentionLangModel
Language model based upon the Encoder units of the transformer. For Theortical back ground please refere to Attention is all you need 
paper @(https://arxiv.org/abs/1706.03762) and for detials regard the impelementation please refere to the source code here and to
google tutorial avilable at https://www.tensorflow.org/beta/tutorials/text/transformer. 


## To DO: 
1- build a pip package for the library. 
2- more documentation and examples

## Notes: 
##### because of the difference in the bifurcating condition between return self-attention weights and outputs and only the output the _fit_ method is not an applicable and a custom training loop should be used

## Current State: 
The Modeler and Annotator Models are ready for deployment.

## Examples: 
from SelfAttentionLangModel.Models import EncoderModels

demoModel=EncoderModels.Modeler(

                                      embedding_dim=16,
                                      vocabulary_size=28,
                                      conditional_string_length=30,
                                      num_encoder_layer=6,
                                      num_heads=4,
                                      num_neuron_pointwise=32,
                                      rate=0.1,\n
                                      return_attent_weights=False
                                         )
                                         


