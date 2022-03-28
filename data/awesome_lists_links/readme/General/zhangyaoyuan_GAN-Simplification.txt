# Sentence Simplification GAN Model
### **Generative Adversarial NetWork (GAN) Using TensorFlow.**

#### Project Introduction
Sentence simplification is a task that produces a simplified sentence from any input sentence. Sentence simplification systems can be useful tools for people whose first language is not English, children, or the elderly to help with understanding. Initial approaches to this task were primarily borrowed from neural machine translation systems with transformations at the sentence level or with architectures based on recurrent neural networks (RNNs).

The core building blocks are a conditional sequence generative adversarial net which comprises of two adversarial sub models : **_Generator_** and **_Discriminator_**. The **Generator** aims to generate sentences which are hard to be discriminated from human-translated sentences; the **Discriminator** makes efforts to discriminate the machine-generated sentences from humantranslated ones.

#### Generator features include:

  - support for advanced RNN architectures:
     - neural machine translation
     - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
     - tied embeddings (Press and Wolf, 2016) https://arxiv.org/abs/1608.05859
     - layer normalisation (Ba et al, 2016) https://arxiv.org/abs/1607.06450
     - mixture of softmaxes (Yang et al., 2017) https://arxiv.org/abs/1711.03953
    
 - training features:
     - multi-GPU support [documentation](doc/multi_gpu_training.md)
     - minimum risk training (MRT)

 - scoring and decoding features:
     - batch decoding
     - scripts for scoring (given parallel corpus) and rescoring (of n-best output)

####Discriminator features include:

 - CNN based Neural Network
 - Policy Gradient Training:
   - Simplicity (SARI Metric)
 	- Grammar (BLUE Metric)
 	- Relevance (Cosine Similarity)
 	



# Dependencies
- NumPy >= 1.11.1
- Tensorflow >= 1.2


# History
- Oct 5, 2018: Basic Functions
- Jan 11, 2019: Major update
- May 22, 2019: Additional update


# Usage Instructions
## **Data Preparation**
**Parallel Dataset Example**: 

Simplified Sentence  
``Apple sauce or applesauce is a puree made of apples.`` 

Complex Sentence  
``Applesauce (or applesauce) is a sauce that is made from stewed or mashed apples.``

**Paper** : _Hwang, William, Hannaneh Hajishirzi, Mari Ostendorf, and Wei Wu. "Aligning sentences from standard wikipedia to simple wikipedia." In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 211-217. 2015._

**Download Dataset** https://cs.pomona.edu/~dkauchak/simplification/data.v2/sentence-aligned.v2.tar.gz


##**Training Step by Step**
#### Generator Part

1. Pretrain Generator part ``sh generaotr.sh``  

2. Generating Negetive dataset ``sh generate_sample.sh``  

#### Discriminator Part

3. Pretrain Discriminator ``sh discriminator_pretrain.sh``  

4. Train The Whole GAN Model ``sh gan_train.sh`` 

## **Configuration Directory**
``GAN-Simplification/configs1``
