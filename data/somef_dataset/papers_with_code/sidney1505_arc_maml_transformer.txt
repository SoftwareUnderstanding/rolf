This model aims to solve the Abstraction and Reasoning Challange (https://www.kaggle.com/c/abstraction-and-reasoning-challenge) with an Universal Transformer model (https://arxiv.org/pdf/1703.03400.pdf), that is trained via model agnostic meta learning (https://arxiv.org/pdf/1807.03819.pdf).

The Transformer model is oriented on the Keras Transformer (https://github.com/kpot/keras-transformer), but some changes were needed to work also for the ARC challange.

In order to run the model you first need to run either "./setup_cpu.sh" or "./setup_gpu.sh" in order to setup the environment the right way.

Furthermore you have to download the data from the challange site, unzip it and place it in a folder "data".

Then you can run "python main.py" in order to execute the model.

Additional information about the usage of the model can be found with "python main.py --help".

If you restart your terminal, please execute "source gpu_env/bin/activate" or "source cpu_env/bin/activate" again.