# The Karl Marx’s Press Review

_A website to visualise a NLP project on text generation with GPT-2_

![made-with-python](https://img.shields.io/badge/Made%20with-Python-E8B90F.svg) ![Maintenance](https://img.shields.io/badge/Maintained%5F-yes-green.svg) ![Website cv.lbesson.qc.to](https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)



### Concept
This website has been developed with Python and Flask to provide a visualisation of **a project on text generation with [GPT-2](https://openai.com/blog/better-language-models/)**. Trained on 8 million webpages, GPT-2 was released in 2019 for generating high-quality short texts based on a few words provided as an initial input. 

For this project, I decided to **fine-tune the GPT-2**, that is to make it especially sensitive to a certain vocabulary or style, so as to reproduce those features while generating text. To achieve this goal and to better be able to assess its results, I first looked for **a corpus of texts featuring a very colourful, if homogenous, rhetoric** to retrain the GPT-2 on. My choice eventually fell on the works of Karl Marx and Friedrich Engels, which I scraped from the website of the [Marx Engels Archive](https://marxists.architexturez.net/archive/marx/index.htm). Moreover, to observe the outcome of “Marxist” textual generation on a variety of topics, I thought it could be interesting to **provide inputs for the GPT-2 from a newspaper**. Hence the idea of this peculiar press review, which is updated with the latest news of [The Guardian](https://www.theguardian.com).
| ![gif](./press-review.gif) |
|:--:|
---
### Architecture
The model is fine-tuned by means of [aitextgen](https://docs.aitextgen.io/), a Python library developed by [Max Woolf](https://github.com/minimaxir). Aitextgen leverages PyTorch to retrain the 124 M version of GPT-2 using the dataset provided by the user. 

The webapp cyclically collects articles from _The Guardian_’s API and uses the language model to generate “Marxist comments” based on them. I also implemented some basic **sentiment analysis** on the generated comments using [VADER](https://github.com/cjhutto/vaderSentiment) (_Valence Aware Dictionary and sEntiment Reasoner_). All these data are eventually stored in the SQL database.

The website also features a function for directly interacting with the model. The Marxist GPT-2 is not _always_ very intelligent, however it is pretty opinionated one, and it is always fun to talk to it! 😉   

| ![gif](./generator.gif) |
|:--:|
|<span style="color:grey"><i>It is a bit long but...the wait is worth the pain!</i></span>|

---
### To Use This Code Locally
#### STEP 1: Generate the dataset and the model:

- Clone this repository
- Go to the folder `data_and_model`:
  - install the requirements with `pip install requirements.txt`;
  - run: `python scraper_preprocesser.py` **to download the dataset on which to fine-tune the GPT-2 model** (`marx.txt`). After running the process, you should see it in a new subfolder called `training_dataset/preprocessed`;
  - **To download and fine-tune the GPT-2 model**, load the Notebook `Text-Generating_GPT-2_Finetuner_on_Colab_GPU.ipynb` into your Google Drive, <u>open it with Google Colaboratory</u> and follow the instructions to create the two files `pytorch_model.bin` and `config.json`;
  - Paste these files into the two subfolders `trained_model` to be found in: `marxist_press_review/article_collector/` and `marxist_press_review/press_review_app/`.

#### STEP 2: Setting the required environment variables

In order for this webapp to work, you will need to set _two environment variables_:
1. The password for the PostgreSQL database which will be created. Unless you do not want to change its name in the `docker-compose.yml` file, this variable must be called `POSTGRES_PASSWORD`;
2. The API key for **The Guardian Open Platform**, which you can generate upon free registration via [this link](https://bonobo.capi.gutools.co.uk/register/developer).  Unless you do not want to change its name in the `docker-compose.yml` file, this variable must be called `GUARDIAN_API_KEY`.

#### STEP 3: Running the webapp with Docker

- Install Docker, then go into the folder `marxist_press_review`:
  - run `docker-compose build` and wait for Docker to set up everything for you;
  - run `docker-compose up` and wait for the log to confirm that the webapp has correctly started (something like `press_review_app_1   | 2021-08-16 15:38:21,336: INFO:  * Running on http://<ANY-ADDRESS-ENDING-BY-:5000/>`);
  - wait for another while, as the software downloads the most recent articles from _The Guardian_'s API to PostgreSQL (the log will confirm that your API key works correctly by printing lines such as: `2021-08-16 15:40:12,819: INFO: Successfully connected to https://content.guardianapis.com/search?section=world: scraping...`);
- open the address `http://localhost:5000` in your browser and the website should appear. 

Have fun talking with Karl Marx!

---

### Further Reading

* Alammar, J. (2019), *The Illustrated GPT-2 (Visualizing Transformer Language Models)*, URL: https://jalammar.github.io/illustrated-gpt2/.
* Radford, A. *et al.* (2019), “Language Models are Unsupervised Multitask Learners”, *OpenAi Blog*, URL: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
* Vaswani, A. *et al.* (2017), “Attention is All You Need”, 31st Conference on Neural Information Processing Systems (*NIPS* 2017), Long Beach (CA), URL: https://arxiv.org/abs/1706.03762.

---
### To Do
- [ ] Increase the size of the training dataset
- [ ] Add more documentation
- [ ] Host on GCP
- [ ] Tests

