# gumbel-softmax-exploration
Exploration of the Gumbel Softmax Paper https://arxiv.org/pdf/1611.01144.pdf

# Requirements
* [docker] https://docs.docker.com/engine/install/

# Run

    git clone git@github.com:stefanthaler/gumbel-softmax-exploration.git
    cd gumbel-softmax-exploration
    sudo docker pull bruthaler/gumbel-softmax-exploration:latest
    bash: sudo docker run -v $(pwd)/:/tf -p 8888:8888 --rm -it --name tfcustom bruthaler/gumbel-softmax-exploration:latest
    fish: sudo docker run -v (pwd)/:/tf -p 8888:8888 --rm -it --name tfcustom bruthaler/gumbel-softmax-exploration:latest
