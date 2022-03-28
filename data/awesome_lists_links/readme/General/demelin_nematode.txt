# NEMATODE: Light-weight NMT toolkit  

**Note: Current implementation is outdated and will be brought up to date soon.**

## Dependencies
- python >= 3.5.2
- tensorflow >= 1.9
- CUDA >= 9.0

## Description
NEMATODE is a light-weight neural machine translation toolkit built around the [transformer](https://arxiv.org/pdf/1706.03762.pdf) model. As the name suggests, it was originally derived from the [Nematus](https://github.com/EdinburghNLP/nematus) toolkit and eventually deviated from Nematus into a stand-alone project, by adopting the transformer model and a custom data serving pipeline. Many of its components (most notably the transformer implementation) were subsequently merged into Nematus.

## Motivation
NEMATODE is maintained with readability and modifiability in mind, and seeks to provide users with an easy to extend sandbox centered around a state-of-the-art NMT model. In this way, we hope to contribute our small part towards facilitating interesting research. Nematode is implemented in TensorFlow and supports useful features such as **dynamic batching**, **multi-GPU training**, **gradient aggregation**, and **checkpoint averaging** which allow for replication of experiments originally conducted on a large number of GPUs on a limited computational budget. 

## Acknowledgements
We would like to thank the authors of the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) libraries for the valuable insights offered by their respective model implementations.

## Caveats
While the core transformer implementation if fully functional, the toolkit continues to be a work-in-progress.

## Performance
On one Nvidia GeForce GTX Titan X (Pascal) GPU with CUDA 9.0, our transformer-BASE implementation achieves the following training speeds:

~4096 tokens per batch, no gradient aggregation, single GPU (effective batch size = ~4096 tokens):
>> 4123.86 tokens/sec

~4096 tokens per batch, gradient aggregation over 2 update steps, 3 GPUs (effective batch size = ~25k tokens):
>> 16336.97 tokens/sec

Following the training regime described in ['Attention is All You Need'](https://arxiv.org/pdf/1706.03762.pdf), our transformer-BASE implementation achieves 27.45 BLEU on the WMT2014 English-to-German task after 148k update steps (measured on newstest2014).

## Use
To train a transformer model, modify the provided example training script - `example_training_script.sh` - as required.

#### Data parameters
| parameter            | description |
|---                   |--- |
| --source_dataset PATH | parallel training corpus (source) |
| --target_dataset PATH | parallel training corpus (target) |
| --dictionaries PATH [PATH ...] | model vocabularies (source & target) |
| --max_vocab_source INT | maximum length of the source vocabulary; unlimited by default (default: -1) |
| --max_vocab_target INT | maximum length of the target vocabulary; unlimited by default (default: -1) |

#### Network parameters
| parameter            | description |
|---                       |--- |
| --model_name MODEL_NAME | model file name (default: nematode_model) |
| --model_type {transformer} | type of the model to be trained / used for inference (default: transformer) |
| --embiggen_model | scales up the model to match the transformer-BIG specifications |
| --embedding_size INT | embedding layer size (default: 512) |
| --num_encoder_layers INT | number of encoder layers (default: 6) |
| --num_decoder_layers INT | number of decoder layers (default: 6) |
| --ffn_hidden_size INT | inner dimensionality of feed-forward sub-layers in FAN models (default: 2048) |
| --hidden_size INT | dimensionality of the model's hidden representations (default: 512) |
| --num_heads INT | number of attention heads used in multi-head attention (default: 8) |
| --untie_decoder_embeddings | untie the decoder embedding matrix from the output projection matrix |
| --untie_enc_dec_embeddings | untie the encoder embedding matrix from the embedding and projection matrices in the decoder |

#### Training parameters
| parameter            | description |
|---                   |--- |
| --max_len INT | maximum sequence length for training and validation (default: 100) |
| --token_batch_size INT | mini-batch size in tokens; set to 0 to use sentence-level batch size (default: 4096) |
| --sentence_batch_size INT | mini-batch size in sentences (default: 64) |
| --maxibatch_size INT | maxi-batch size (number of mini-batches sorted by length) (default: 20) |
| --max_epochs INT | maximum number of training epochs (default: 100)
| --max_updates INT | maximum number of updates (default: 1000000)
| --warmup_steps INT | number of initial updates during which the learning rate is increased linearly during learning rate scheduling (default: 4000) |
| --learning_rate FLOAT | initial learning rate (default: 0.0002) (DOES NOTHING FOR NOW) |
| --adam_beta1 FLOAT | exponential decay rate of the mean estimate (default: 0.9) |
| --adam_beta2 FLOAT | exponential decay rate of the variance estimate (default: 0.98) |
| --adam_epsilon FLOAT | prevents division-by-zero (default: 1e-09) |
| --dropout_embeddings FLOAT | dropout applied to sums of word embeddings and positional encodings (default: 0.1) |
| --dropout_residual FLOAT | dropout applied to residual connections (default: 0.1) |
| --dropout_relu FLOAT | dropout applied to the internal activation of the feed-forward sub-layers (default: 0.1) |
| --dropout_attn FLOAT | dropout applied to attention weights (default: 0.1) |
| --label_smoothing_discount FLOAT | discount factor for regularization via label smoothing (default: 0.1) |
| --grad_norm_threshold FLOAT | gradient clipping threshold - may improve training stability (default: 0.0) |
| --teacher_forcing_off | disable teacher-forcing during model training (DOES NOTHING FOR NOW) |
| --scheduled_sampling | enable scheduled sampling to mitigate exposure bias during model training (DOES NOTHING FOR NOW) |
| --save_freq INT | save frequency (default: 4000) |
| --save_to PATH | model checkpoint location (default: model) |
| --reload PATH | load existing model from this path; set to 'latest_checkpoint' to reload the latest checkpoint found in the --save_to directory |
| --max_checkpoints INT | number of checkpoints to keep (default: 10) |
| --summary_dir PATH | directory for saving summaries (default: same as --save_to) |
| --summary_freq INT | summary writing frequency; 0 disables summaries (default: 100) |
| --num_gpus INT | number of GPUs to be used by the system; no GPUs are used by default (default: 0) |
| --log_file PATH | log file location (default: None) |
| --debug | enable the TF debugger |
| --gradient_delay INT | number of steps by which the optimizer updates are to be delayed; longer delays correspond to larger effective batch sizes (default: 0) |
| --track_grad_rates | track gradient norm rates and parameter-grad rates as TensorBoard summaries |

#### Development parameters
| parameter            | description |
|---                   |--- |
| --valid_source_dataset PATH | source validation corpus (default: None) |
| --valid_target_dataset PATH | target validation corpus (default: None) |
| --valid_freq INT | validation frequency (default: 4000) |
| --patience INT | number of steps without validation-loss improvement required for early stopping; disabled by default (default: -1) |
| --validate_only | perform external validation with a pre-trained model |
| --bleu_script PATH | path to the external validation script (default: None); receives path of translation source file; must write a single score to STDOUT. |

#### Reporting parameters
| parameter            | description |
|---                   |--- |
| --disp_freq INT | training metrics display frequency (default: 100) |
|  --greedy_freq INT | greedy sampling frequency (default: 1000) |
|  --sample_freq INT | weighted sampling frequency; disabled by default (default: 0) |
|  --beam_freq INT | beam search sampling frequency (default: 10000) |
|  --beam_size INT | size of the decoding beam (default: 4) |

#### Translation parameters
| parameter            | description |
|---                   |--- |
| --translate_only | translate a specified corpus using a pre-trained model |
| --translate_source_file PATH | corpus to be translated; must be pre-processed |
| --translate_target_file PATH | translation destination |
| --translate_with_beam_search | translate using beam search |
| --length_normalization_alpha FLOAT | adjusts the severity of length penalty during beam decoding (default: 0.6) |
| --no_normalize | disable length normalization |
| --full_beam | return all translation hypotheses within the beam |
| --translation_max_len INT | Maximum length of translation output sentence (default: 100) |


## Citation
If you decide to use NEMATODE in your work, please provide a link to this repository in the corresponding documentation.

## TODO
1. Update code
