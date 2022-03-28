# IPO Prediction in With Graph Convolutional Neural Network

This repository is a demo to show the process of forming a data science project around data that is stored in a TigerGraph graph database. It uses Giraffle with Gradle to install GSQL queries onto the database instance, and then call those REST endpoints with pyTigerGraph, a Python wrapper written to make accessing TigerGraph databases easy. We then feed the data that we recieve from the database into a Graph Convolutional Neural Network. Due to various simplifications and computational constraints, the GCN doesn't perform that great in classifying which companies will IPO or not, but it does serve as a good demo of what can be done with other datasets.

## Setup TigerGraph Cloud Instance
To setup your own version of the server, head over to [tgcloud.io](https://tgcloud.io/) and create a free account. It will walk you through creating your own cloud database instance, but make sure to choose the CrunchBase knowledge graph as the template to get started with.

## Getting Gradle Installed and Setup
Follow the directions [here](https://gradle.org/install/) to get Gradle installed on your machine. Once that is done, you will need to get an SSL certification from your cloud instance. To do this, run:
```bash
openssl s_client -connect hostname.i.tgcloud.io:14240 < /dev/null 2> /dev/null | \
openssl x509 -text > cert.txt
```
## Installing the Python Packages Needed
You need to install quite a few Python packages. Good news, is that this is easily done via pip:
```
pip3 install pytigergraph
pip3 install torch
pip3 install dgl
pip3 install networkx
```
You are now all ready to try out the notebook. Find it in /py_scripts.

## Credits
<p><img alt="Picture of Parker Erickson" height="150px" src="https://avatars1.githubusercontent.com/u/9616171?s=460&v=4" align="right" hspace="20px" vspace="20px"></p>

Demo/tutorial written by Parker Erickson, a student at the University of Minnesota pursuing a B.S. in Computer Science. His interests include graph databases, machine learning, travelling, playing the saxophone, and watching Minnesota Twins baseball. Feel free to reach out! Find him on:

* LinkedIn: [https://www.linkedin.com/in/parker-erickson/](https://www.linkedin.com/in/parker-erickson/)
* GitHub: [https://github.com/parkererickson](https://github.com/parkererickson)
* Medium: [https://medium.com/@parker.erickson](https://medium.com/@parker.erickson)
* Email: [parker.erickson30@gmail.com](parker.erickson30@gmail.com)
----
GCN Resources:
* DGL Documentation: [https://docs.dgl.ai/](https://docs.dgl.ai/)
* GCN paper by Kipf and Welling [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
* R-GCN paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
---- 
Notebook adapted from: [https://docs.dgl.ai/en/latest/tutorials/basics/1_first.html](https://docs.dgl.ai/en/latest/tutorials/basics/1_first.html)