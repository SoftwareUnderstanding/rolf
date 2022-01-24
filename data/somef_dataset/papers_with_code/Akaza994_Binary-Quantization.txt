# Binary-Quantization
Implementation of Binary quantization of ResNet. Code used in work: https://arxiv.org/abs/2008.13305


Trained models:
|    Model   | Dataset  | Channel Sparsity | Loss |
|------------|----------|------------------|------|
|En1ResNet56 |  Cifar10 |   33.18%         |  (9) |
|En1ResNet110|  Cifar100|   23.93%         |  (9) |
|En2ResNet56 |  Cifar100|   43.94%         |  (9) |
|En2ResNet56 |  SVHN    |   49.03%         |  (9) |


References:

Binary Connect: https://arxiv.org/abs/1511.00363

Binary Relax: https://arxiv.org/abs/1801.06313

Binary quatization projection: Algorithm 1 in https://arxiv.org/pdf/1603.05279.pdf.


