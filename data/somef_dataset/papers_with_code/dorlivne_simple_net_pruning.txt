# simple_net_pruning
pruning simplenet

this repo is about training and pruning Simple-Net
an explanation can be found here:
https://arxiv.org/abs/1608.06037
there are 3 stages:
1) train.py
    this script trains the simplenet on cifar 10 data base, if you train with the defualt HP it will get to 90% ( give or take 1%) 
    test accuracy.
    use --help when initiating the script to see easy to change parameters
    note: the learning rate is defined in configs.py has a static method which is correlated to the number of epoch passed
2) prune.py
    this script prunes the trained model from train.py and outputs a pruned model, there are some HP which can be explained here:
    https://arxiv.org/pdf/1710.01878.pdf 
    and here 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/model_pruning/README.md
    use --help to learn more
    you can always use the defualt parameters just be careful with the sparsity_goal which defines the output model sparsity level
3) create_sparse.py
    this script creates a sparse model from the pruned model, basiclly it tears down all the pruning nodes from the graph and
    create a pb model of the sparse model.
    
Thank you.
    
