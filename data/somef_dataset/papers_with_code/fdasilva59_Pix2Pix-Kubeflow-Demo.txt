# Pix2Pix : Aerial images to maps

This repository contains my Pix2Pix implementation that was used for a Kubeflow Pipelines demo at Google Next 2019


## Why this project ?

There were several goals behind this project:
- Experimenting with image translations using GAN/Pix2Pix and Tensorflow
- "Benchmarking" a GPU datalab server at work
- Experimenting and developing expertise with the Kubeflow ecosystem (+sharing the results and promoting Kubeflow inside our organization at work)

When I first started developping this project, I focused a lot on the early stage of a Data Science project, when a Data Scientist wants  to test new ideas without having to care where the code will be executed, and without any needs to manually build docker containers to package the code. Since then, this is even more easier with Kubeflow Fairing.

The main reasons why you should use Kubeflow are :
- It allows easy, repeatable, portable deployments on a diverse infrastructure (laptop <-> ML rig <-> training cluster <-> Production cluster)
- It makes the work easier and more efficient. Data Scientists can iterate faster
- In the end, Data Scientists don't have to care about setting up the environment where the code will be executed
   - ’On-prem” clusters resources are not unlimited, and not always available. 
   - Data Scientists don’t want to spend time configuring some architecture to run their experiments.
   - One of the best features of  using GKE and Kubeflow together, it that you can setup your GKE Cluster with autoscaling. For example, you can add a GPU node-pools to your Kubeflow cluster, that scale on-demand GPU nodes from 0 to whatever value you want.
   - This type of hybrid configuration makes the work very efficient, and cost-efficient, as you can iterate faster, while paying only for the extra GKE resources you need to expand an « on-prem » cluster/workstation.


## Acknowledgments

Pix2Pix work based on : "**Image-to-Image Translation with Conditional Adversarial Networks**" 

See : [arXiv:1611.07004v3 [cs.CV]](https://arxiv.org/abs/1611.07004) by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros - [Project Homepage](https://github.com/phillipi/pix2pix) 


You might also be interested in more recent works:

- [Pix2Pix Tensorflow 2.0 tutorial](https://www.tensorflow.org/beta/tutorials/generative/pix2pix)
- [Pix2Pix HD](https://tcwang0509.github.io/pix2pixHD/) "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"

The maps dataset used contain 1096 training images scraped from Google Maps



## Setup

Follow the [setup instructions](./SETUP.md)


### Requirements

This demo requires a Google Cloud Platform (GCP) project with an IAP-enabled cluster running on Kubernetes Engine (GKE). The Pix2Pix code requires Python 3.5, Tensorflow 1.12 and Kubeflow 0.5.0.


### Setup overview
- Prerequisites : Have a GCP account with billing enabled
- Create a GCP project (don't forget to adjust the GPU quotas for the considered region)
- An endpoint protected by GCP IAP will be created for accessing kubeflow. Follow [these instructions](https://www.kubeflow.org/docs/gke/deploy/oauth-setup/) to create an OAuth client and then enter as IAP Oauth Client ID and Secret
- Create a Kubeflow cluster on GKE, using the [Google Cloud Deploy tool](https://deploy.kubeflow.cloud/#/deploy)
- Perform additional Setup for REMOTE Kubeflow pipeline execution 
- Create GCP service account with the necessary permissons, and added as an 'IAP-secured Web App User' 
- Create a storage bucket in Google Storage
- Add an NFS server running in GKE inside the Kubeflow namespace
- Create a GPU node pool with Autoscaling enabled in GKE

### Known issues 

June/July 2019 : It seems that there are some issues with Kubeflow 0.5 when using IAP. In particular I have noticed some difficulties to access Jupyter Notebooks or spawning a Jupyter server that mounts a NFS volume. Until the issue is solved, a bypass solution is provided in order to avoid using IAP.

### Next steps / Work in progress

- Include the instructions to setup the cluster for REMOTE Kubeflow pipeline execution (instead of external reference)
- Rework the GKE cluster setup for deployment automation (DevOps for MLOps ! )
- Investigate Pix2Pix HD
- Use a cleaner dataset (without traffic information...)
- Include a Kubeflow Fairing version


## How to use

Setup your local environment and your GKE cluster, clone the repository or import the source code, and execute the notebooks ! 

**Don't forger to customize in the Notebooks your bucket URL, Kubeflow cluster endpoints and client_id**


- `1-Pix2Pix-local.ipynb ` is expected to be executed locally in a Jupyter notebook running on anykind of GPU server (without Kubeflow)
- `2-Pix2Pix-KFP.ipynb ` is expected to be executed inside the Kubeflow's Jupyter notebook running on a GKE cluster (with some GPU nodes pool available).
- `3-Pix2Pix-KFP-REMOTE.ipynb ` is expected to be executed in a local Jupyter notebook instance (for instance running on a laptop without GPU), and it will interact with a Kubeflow cluster running on GKE


### Some results 

Results after ~200 steps (200 epochs with batch size of 1) :

From Left to Right : **Source Image / Generated ("Translated") Image / Target Image (ground truth)**

![Source Image](doc-assets/img_a-02-174658.png "Source Image")
![Generated Image](doc-assets/fake_b-02-174658.png "Generated Image")
![Target Image (ground truth)](doc-assets/img_b-02-174658.png "Target Image")

<br>


![Source Image](doc-assets/img_a-02-174852.png "Source Image")
![Generated Image](doc-assets/fake_b-02-174852.png "Generated Image")
![Target Image (ground truth)](doc-assets/img_b-02-174852.png "Target Image")

<br>


![Source Image](doc-assets/img_a-02-173920.png "Source Image")
![Generated Image](doc-assets/fake_b-02-173920.png "Generated Image")
![Target Image (ground truth)](doc-assets/img_b-02-173920.png "Target Image")


### Remark on the implementation

In order to avoid some kind of "noise artifacts" to appears in the generated images, and while it was not specified in the Author's paper or original Torch code, I have added some gradient clipping during the optimization steps which solved the problem immediately. 

Example of "Noise artifacts" that were appearing in the generated images before adding Gradient Clipping :

![Noisy Image](doc-assets/fake_b-13-222626.png "Noisy Image")



## Kubeflow Pipelines at Google Next 2019

See Google Next 2019 Breakout session : [ML Ops Best Practices on Google Cloud](https://cloud.withgoogle.com/next/sf/speakers?session=MLAI101) (Apr 2019)

> Creating an ML model is just a starting point. To bring it into production, you need to solve various real-world issues, such as building a pipeline for continuous training, automated validation of the model, scalable serving infrastructure, and supporting multiple environments in increasingly common hybrid and multi-cloud setups. In this session, we will learn the concept of "ML Ops" (DevOps for ML) and how to leverage various Google initiatives like TFX, Kubeflow Fairing (Hybrid ML SDK) and Kubeflow Pipelines to build and maintain production quality ML systems.

Speakers :

- Kaz Sato, Developer Advocateat Google Cloud
- Zia Syed Sr. Engineering Manager at Google Cloud
- Robin Zondag, Global Head of Atos AI/ML Labs
- Fabien Da Silva, Artificial Intelligence Expert at Atos/Worldline


Thanks to the Google Team for all their support during the preparation of Google Next event (Special thanks to Amy Unruh for the great support and tutorials !)

