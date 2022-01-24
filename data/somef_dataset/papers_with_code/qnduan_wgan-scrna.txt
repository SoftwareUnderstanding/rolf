# Gan implemented using pytorch

2d-mixtures:
An implementation of the original wgan algorithm. The parameters and utility functions are copied from
https://github.com/musyoku/wasserstein-gan
It just worked as reported.

2d-mixtures-improved:
An implementation of the improved wgan algorithm using gradient norm penalty as described in this paper:
https://arxiv.org/abs/1704.00028

Initially, I used lambda=10 as suggested by the author. The gan network behaved like a standard gan with mode collapse. Tuning the optimizer parameters did not mitigate the problem. Then I changed lambda to 1 and it worked very nicely. The results are much better than the original wgan.
