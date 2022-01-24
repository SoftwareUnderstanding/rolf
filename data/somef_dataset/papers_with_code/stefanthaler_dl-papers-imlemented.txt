# Deep learning papers implemented

Implementations of deep learning papers. z  

# Run

1) Get environment
In [Fish](https://fishshell.com/):

    sudo docker build -t <maintainer>/<experiment_name> .
    sudo docker run -u (id -u):(id -g) -v (pwd)/:/tf -p 8888:8888 --rm -it --name <container_name> <maintainer>/<experiment_name>:latest

In Bash:

    sudo docker build -t <maintainer>/<experiment_name> .
    sudo docker run -u $(id -u):$(id -g) -v ($pwd)/:/tf -p 8888:8888 --rm -it --name <container_name> <maintainer>/<experiment_name>:latest

2) Open the url in the browser
3) Click runall

## Interested
Papers I will eventually try to implement
* Variational Autoencoder 
* Neural turing machines
* Wordvectors
* Sampled softmax 
* Negative contrastive estimation
* Triplet Networks 
* Siamese Networks 

## 2016
### Gumbel Softmax
* https://arxiv.org/pdf/1611.01144.pdf
