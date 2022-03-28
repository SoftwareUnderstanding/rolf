# BERT word embedding in tf.keras

## Info

Author: Márton Torner

This work is based on the code:
- in the article: https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
- in the official repo: https://github.com/google-research/bert

BERT paper: https://arxiv.org/pdf/1810.04805.pdf

A very good article to understand BERT: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

## Content

This repo contains an implementation for BERT word embeddings as a tf.keras layer, a preprocessor 
to generate the proper inputs from an array of sequences and also a simple BertEncoder to calculate Bert embeddings 
out-of-the-box. A dockerfile is also provided to set up a proper environment.

## Docker

### Build image

```bash
docker build -t {USERNAME}/bert-keras -f docker/Dockerfile .
```

### Run container

Starts the jupyter notebook server, tensorboard and ssh server. Sometimes Ctrl+P+Q does not work but closing the
terminal window is evenly good (will not kill the notebook process).

```bash
docker run --runtime=nvidia -e LOCAL_USER_ID=`id -u $USER` -e LOCAL_GROUP_ID=`id -g $USER` --name 'bert-keras' -p xxxx:8888 -p yyyy:6006 -p zzzz:22 -v {bert-keras root folder}:/app {USERNAME}/bert-keras:latest
```

#### Get notebook token

```bash
docker exec -it {name} bash -c "cat /var/log/supervisor/jupyter-notebook-stderr*" | grep token
```

#### Password

The user and root password in the container: `password123456`. Please change it after creating the container.
