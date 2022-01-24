# BREVIS-News 2.0

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge&color=339933)](http://makeapullrequest.com)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?&style=for-the-badge&color=FFD43B)](https://lbesson.mit-license.org/)

Original unfinished project [Brevis-News](https://github.com/niladridutt/Brevis-News)

* **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**: https://arxiv.org/abs/1910.13461

# Current Status:
## Completed Features:
- [x] Created custom serving model based on BART, pre-trained model from huggingface and fine-tuned on inshorts data
- [x] Created Django backend for:
  - [x] Article scraping 
  - [x] Google-News API trending news feed
- [x] Single line docker-compose deployment
- [x] Designing front-end 


## Pending Work:
- [ ] Overhaul front-end 
- [ ] Fine-Tune model on more dataset
- [ ] Heroku Deployment
- [ ] Trending news auto-summarization

# Usage
## With docker-compose
Just run docker compose up and the everything should just work.
* Does not work on windows due to WSL not supporting host mode on the port
```bash
docker-compose up
```
It will take a long time for the first run

## Without docker-compose
### BART Serving:
#### Step 1: Clone this repo

```bash
git clone https://github.com/vinayak19th/MDD_Project
```

#### Step 2: pull the docker container

> For information on installing docker, check [here](https://docs.docker.com/engine/install/)

```bash
docker pull vinayak1998th/bart_serve:1.0
```
#### Step 3: Launch the container

##### CPU Runtime
```bash
docker run -d -p 8501:8501 -p 8500:8500 --name bart vinayak1998th/bart_serve:cpu
```
If you have an NVIDA CUDA supported GPU, you can run the server for GPU runtime
##### GPU Runtime
```bash
docker run --runtime=nvidia -d -p 8501:8501 -p 8500:8500 --name bart vinayak1998th/bart_serve:gpu
```
#### Step 4 : Test

The code for testing after the server is running is in the [Serving_Test](./Serving_Test.ipynb) notebook

### Contributing to this Project
Read Contributing.md file in the repository to learn more about making pull requests, and contributing in general.

## Project Maintainers
The project is currently being maintained by :  
1) [Vinayak Sharma](https://github.com/vinayak19th) 
