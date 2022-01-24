[EfficientNetV2-S](https://arxiv.org/abs/2104.00298) implementation using PyTorch

#### Steps
* Configure `imagenet` path by changing `data_dir` in `train.py`
* `python main.py --benchmark` for model information
* `python -m torch.distributed.launch --nproc_per_node=$ main.py --train` for training model, `$` is number of GPUs
* `python main.py --test` for testing, `python main.py --test --tf` for ported weights testing

#### Note
* The model achieved 82.7 top-1 after 150 epochs
* The model ported from original TensorFlow showed 83.8 top-1
```
Number of parameters: 23941296
Time per operator type:
        778.049 ms.    70.6258%. Conv
        255.227 ms.    23.1677%. Sigmoid
          56.91 ms.    5.16589%. Mul
         6.1573 ms.   0.558916%. Add
        4.69289 ms.   0.425987%. ReduceMean
       0.613303 ms.  0.0556713%. FC
        1101.65 ms in Total
FLOP per operator type:
         17.277 GFLOP.    99.7074%. Conv
      0.0419251 GFLOP.   0.241954%. Mul
     0.00519322 GFLOP.  0.0299706%. Add
       0.003585 GFLOP.  0.0206894%. FC
        17.3277 GFLOP in Total
Feature Memory Read per operator type:
        295.875 MB.    50.5134%. Mul
        241.136 MB.     41.168%. Conv
        41.5457 MB.     7.0929%. Add
        7.17917 MB.    1.22567%. FC
        585.737 MB in Total
Feature Memory Written per operator type:
          167.7 MB.    49.2361%. Mul
        152.127 MB.    44.6639%. Conv
        20.7729 MB.    6.09882%. Add
          0.004 MB. 0.00117438%. FC
        340.605 MB in Total
Parameter Memory per operator type:
        87.8034 MB.    92.4486%. Conv
          7.172 MB.    7.55143%. FC
              0 MB.          0%. Add
              0 MB.          0%. Mul
        94.9754 MB in Total
```

