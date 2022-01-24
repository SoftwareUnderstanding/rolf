# ResNet-VAE
Code for NIPS paper of the work of the PGM course project.

The main code is in `codes/flow_vae_tf.py`.

* To run Vanilla VAE, run
```
python flow_vae_tf.py --exp_name [name of the experiment for logging] 
```

* To run Vanilla planar normalizing flow (https://arxiv.org/pdf/1505.05770.pdf), run
```
python flow_vae_tf.py --flow planar --exp_name [name of the experiment for logging] 
```

* To run Vanilla ResNet flow (ours), run
```
python flow_vae_tf.py --flow resnet --exp_name [name of the experiment for logging] 
```

Paramters for each of the flows can be modified inside `codes/flow_vae_tf.py`.

# Logging
Tensorboard logs are automatically generated in `./logs/` folder. Run 
```
tensorboard --logdir . 
```

# Contact
* Hadi Salman `hadicsalman at gmail dot com`
* Aditya Kiran `rchavali at andrew dot cmu dot edu`
* Naman Gupta `namang at andrew dot cmu dot edu`
* Kayhan batmanghelich `batmanghelich at gmail dot com`
