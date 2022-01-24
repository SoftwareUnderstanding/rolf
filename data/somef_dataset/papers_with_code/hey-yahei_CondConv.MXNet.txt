# CondConv    
Reproduce work in [arXiv:1904.04971v2](https://arxiv.org/abs/1904.04971v2) with the implement of MXNet-Gluon.     

## Do CondConv with grouped convolution         
I use groupwise convolution to implement CondConv easily --    
1. Combine kernels then do convolution        
    1. Reshape `x` from `(bs, c, h, w)` to `(1, bs*c, h, w)`     
    2. Combine `weight` from `(k, oc, c, kh, kw)` to `(bs, oc, c, kh, kw)` and then reshape to `(bs*oc, c, kh, kw)`
    3. Combine `bias` from `(k, oc)` to `(bs, oc)` and then reshape to `(bs*oc, )`
    4. Do convolution with `num_filter=bs*oc` and `num_group=bs` and get outputs with shape `(1, bs*oc, oh, ow)`
    5. Reshape outputs to `(bs, oc, oh, ow)` which are the final results for CondConv
2. Do convolution then combine outputs     
    1. Tile `x` on the second axis for `k` times, and get a new `x` with shape `(bs, k*c, h, w)`
    2. Reshape `weight` from `(k, oc, c, kh, kw)` to `(k*oc, c, kh, kw)`
    3. Reshape `bias` from `(k, oc)` to `(k*oc, )`
    4. Do convolution with `num_filter=k*oc` and `num_group=k` and get outputs with shape `(bs, k*oc, oh, ow)`
    5. Reshape outputs to `(bs, k, oc, oh, ow)` and combine to `(bs, oc, oh, ow)` which are the final results for CondConv

For small `k`(<8), training with latter method is faster.   
For large `k`(>=8), training with the former method is suggested.

## Experiment on cifar_resnet20_v1      

| num_experts | Parameters | FLOPS | Top-1 Acc |
|:---:|:---:|:---:|:---:|
|(baseline)|274,042|41,013,878|91.51%|
|4|1,078,402(+293%)|42,087,854(+2.6%)|91.77%|
|8|2,150,026(+684%)|43,161,830(+5.2%)|91.81%|
|16|4,293,274(+1467%)|45,309,782(+10.5%)|91.89%|
|32|8,579,770(+3031%)|49,605,686(+20.9%)|92.26%|
|(resnet56)|860,026(+314%)|126,292,598(+308%)|92.85%|

-----------         
More details refer to [CondConv：按需定制的卷积权重 | Hey~YaHei!](https://www.yuque.com/yahei/hey-yahei/condconv)