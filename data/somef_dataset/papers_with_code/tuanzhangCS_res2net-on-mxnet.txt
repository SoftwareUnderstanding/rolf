# res2net-on-mxnet
**Try to reproduce res2net using mxnet**  
res2net https://arxiv.org/abs/1904.01169v1  
I'm training res2net on cifar10 now.

## Some problem
When I train the network by using mx.mod.Module and it's fit function, after 200 epoch, the val accuracy only achieve 0.88, at the same time train accuracy is 0.99.
But when I replace Module with mx.model.FeedForward and mx.model's fit function, after 200 epoch, the val acc can achieve 0.92 and train acc is 0.99 or 1.0.

I try to add batch-norm and activation to get better model...
