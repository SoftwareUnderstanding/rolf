# Human Protein Atlas Challenge

This is an implementation and application of the SENet-154 neural network architecture (https://arxiv.org/pdf/1709.01507.pdf) to the Human Protein Atlas Challenge dataset on Kaggle. 

Performance on the leaderboard: Top 12% (234/2172)

This is a record of the various iterations and experiments I conducted along the way to create a convolutional neural network that to recognise proteins found on microscopy staining images. 

Used the FastAI library as well as the pretrained models from: https://github.com/Cadene/pretrained-models.pytorch.

If you are considering using some of this code, things to note:
1. I used tabbed views to retain results of previous experiments - this means that trying to read the notebooks in order may be confusing
2. The code using the FastAI library is likely to be out of date (I used v0.7 whereas v1.0 has been launched)
3. The one-cycle policy did not seem to work well and I had trouble tuning the hyperparameters as recommended (https://sgugger.github.io/the-1cycle-policy.html)  
