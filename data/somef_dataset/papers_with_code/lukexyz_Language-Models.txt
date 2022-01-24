# ULMFiT NLP Transfer Learning :earth_africa::book::speech_balloon:
Sentiment analysis via prediction of restaurant reviews using `ULMFiT (2018)`, a state-of-the-art method (for 2018) which provides a framework for NLP transfer learning. (https://arxiv.org/abs/1801.06146)

To build the text classification model, there are three stages:  

1. :earth_africa: **General-Domain LM Pretraining**  
A pretrained `AWD-LSTM SequentialRNN` is imported, which works as a sequence generator (i.e. predicts the next word) for a general-domain corpus, in our case the `WikiText103` dataset.

2. :book: **Target Task LM Fine-Tuning**  
The `AWD-LSTM Language Model` is fine-tuned on the domain-specific corpus (Yelp reviews), to be able to generate fake restaurant reviews.

3. :speech_balloon: **Target Task Classifier**  
The embeddings learnt from these first two steps are imported into a new `classifier model`, which is then fine-tuned on the target task (star ratings) with gradual unfreezing of the final layers.

<p align="center" >
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/Artboard%201@1.5x.png?raw=true">
</p>

  → :notebook_with_decorative_cover: See [ULMFiT-Yelp.ipynb](notebooks/02-ULMFiT-Yelp-Full-Train.ipynb) for notebook 
  
  → :page_with_curl: See [arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146) for paper 

<br/>

## Synthetic Text Generation
After stage 2 of the process is complete, the `AWD-LSTM` RNN language model can now be used for synthetic text generation. The original RNN model was trained to predict the next word in the `WikiText103` dataset, and we have fine-tuned this with our yelp corpus to predict the next word in a restaurant review.

```python
learn.predict("I really loved the restaurant, the food was")
```
> I really loved the restaurant, the food was `authentic`

```python
learn.predict("I hated the restaurant, the food tasted")
```
> I hated the restaurant, the food tasted `bad`

You can generate reviews of any length. The output generally has a believable sentence structure, but they tend to lack higher-order coherency within a paragraph. This is because the RNN has no memory of the start of the sentence by the time it reaches the end of it. Larger `transformer` attention models like OpenAI GPT-2 or BERT do a better job at this.
```python
learn.predict("The food is good and the staff", words=30, temperature=0.75)
```
> The food is good and the staff is very friendly. We had the full menu and the Big Lots of Vegas. The food was ok, but there was nothing and this isn't a Chinese place.


## Classifier: Predicting the Star-value of a Review ★★★★★
The overall accuracy of the trained classifier was `0.665`, which means that giving the model and un-seen restaurant review it can predict its rating (1-5 stars) correctly `66.5%` of the time.

_Examples_  

Prediction: 5  | Actual: 5  
`(INPUT 25816) You can count on excellent quality and fresh baked goods daily. The patisseries are refined and always delicious. I am addicted to their home made salads and strong coffee. \nYou can order customized cakes and impress your guests. Everything here is made with the finest ingredients. It never disappoints. \n\nThe service is formal. You are always treated with respect. Sometimes I don't mind when they call me Madame but I always correct them and ask to be called \"Mademoiselle, SVP!\"\n\nI guarantee you will return here many times.`  

Prediction: 4  | Actual: 3  
`(INPUT 28342) 8 of us just finished eating here.  Service was very friendly, prices were definitely reasonable, and we all really enjoyed our meals. \n\nI would come back again for sure!\n\nUnfortunately I didn't snap any photos of our food, but here are a few of the place.`  

Prediction: 2  | Actual: 2  
`(INPUT 43756) The food was not all that.  The customer service was just okay. Don't get what all the rave is about??`

## Results
Plotting an Actual vs. Predicted matrix gives us a visual representation of the accuracy of the model. True positives are highlighted on the diagonal. So even when it makes the prediction wrong - the error usually is only off by only 1 star. 
<p align="center">
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/actual_vs_predicted.png?raw=true" width="350">
</p>
<br/>


## Improvements 
In the paper [MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761) (2019), the transfer learning language model is improved using  
1. `Subword Tokenization`, which uses a mixture of character, subword and word tokens, depending on how common they are. These properties allow it to fit much better to multilingual models (non-english languages).
    
<p align="center">
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/multifit_vocabularies.png?raw=true" width="400">
</p>
<br/>

2. Updates the `AWD-LSTM` base RNN network with a `Quasi-Recurrent Neural Network` (QRNN). The QRNN benefits from attributes from both a CNN and an LSTM:
* It can be parallelized across time and minibatch dimensions like a CNN (for performance boost) 
* It retains the LSTM’s sequential bias (the output depends on the order of elements in the sequence).  
    `"In our experiments, we obtain a 2-3x speed-up during training using QRNNs"`

<p align="center" >
  <img src="https://github.com/lukexyz/Language-Models/blob/master/img/multifit_qrnn.png?raw=true" width="550">
</p>

> _"We find that our monolingual language models fine-tuned only on `100 labeled examples` of the corresponding task in the target language outperform zero-shot inference (trained on `1000 examples` in the source language) with multilingual BERT and LASER. MultiFit also outperforms the other methods when all models are fine-tuned on 1000 target language examples."_

Reference: `Efficient multi-lingual language model fine-tuning` by Sebastian Ruder and Julian Eisenschlos (http://nlp.fast.ai/classification/2019/09/10/multifit.html) 



## Installation on AWS
`Deep Learning AMI (Ubuntu 16.04) Version 25.3`, GPU `p2.xlarge` for training :ballot_box_with_check:, `120 GB`

##### SSH into new linux box, create conda environment
    $ ssh -i "<key>.pem" ubuntu@ec2-<public-ip>.us-east-2.compute.amazonaws.com
    $ conda create -n fastai python=3.7
    $ conda activate fastai

##### Dependencies
    $ conda install jupyter notebook -y
    $ conda install -c conda-forge jupyter_contrib_nbextensions
    $ conda install fastai pytorch=1.0.0 -c fastai -c pytorch -c conda-forge

##### Update jupyter kernels (optional)
    $ conda install nb_conda_kernels -y
    $ python -m ipykernel install --user --name fastai --display-name "fastai v1"
    $ conda install ipywidgets

##### Validate GPU installation
    $ python -m fastai.utils.show_install

##### Run notebooks
    $ jupyter notebook --ip=0.0.0.0 --no-browser
    # http://<public IP>:8888/?token=<token>

##### Acknowledgements
* [A Code-First Introduction to NLP](https://github.com/fastai/course-nlp)
* [Universal Language Model Fine-Tuning (ULMFiT)](https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit)  
* [NLP & fastai | MultiFiT](https://mc.ai/nlp-fastai-multifit/)
