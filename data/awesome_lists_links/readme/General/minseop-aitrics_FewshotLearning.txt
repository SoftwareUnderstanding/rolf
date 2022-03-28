# FewshotLearning

In this repo, we reproduced some few-shot learning methods which are 

1. Prototypical Network for Few-shot Learning (https://arxiv.org/abs/1703.05175)
2. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (https://arxiv.org/abs/1703.03400)
3. Neural Statistician (in future)
...

We tested each methods with miniImagenet dataset, which can be found in https://github.com/renmengye/few-shot-ssl-public

# how to use
First, download miniImagenet data and run generate_dataset_from_pkl.py with dataroot

python generate_dataset_from_pkl --datadir "downloaded pkl dir"

Then, you can run script file to train the model. 

# results
MAML

5way-1shot : 48.564 (0.840)  / reported 48.70 (1.84)

5way-5shot : 64.127 (0.721)  /  reported 63.11 (0.91)

Protonet

5way-1shot : 52.547 (0.766)  / reported 49.42 (0.78)

5way-5shot : 67.673 (0.648)  /  reported 68.20 (0.66)



-------other dataset (cy...)
MAML 
5way-1shot : 48.036 (0.820)
5way-5shot : 64.460 (0.711)

Prototnet

5way-1shot : 50.556 (0.834)
5way-5shot : 66.747 (0.659)


