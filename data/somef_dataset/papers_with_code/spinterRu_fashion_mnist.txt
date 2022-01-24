# fashion_mnist

```
(CONV3x3)x3 -> MAXPOOL -> (CONV3x3)x2 -> MAXPOOL -> FC(2048) -> FC(512) -> FC(256) -> SOFTMAX(10)
Test accuracy: 0.94769996 (1024 neurons in the first FC layer, BN after RELU)
Test accuracy: 0.9482 (2048 neurons in the first FC layer, BN after RELU)
Test accuracy: 0.9498 (2048 neurons in the first FC layer, BN BEFORE RELU)

LeakyRelu (https://arxiv.org/abs/1505.00853)
BatchNorm before non-linearity(Leaky-RELU): 0.9466
BatchNorm after non-linearity (Leaky-RELU): 0.9448999
Test accuracy: 0.95009995 (PReLU only in FC)
Test accuracy: 0.949 (LeakyRELU only in FC)

(CONV3x3)x3 -> MAXPOOL -> (CONV3x3)x2 -> CONV1x1 -> MAXPOOL -> DROPOUT -> FC(2048)  DROPOUT -> FC(512) -> DROPOUT -> FC(256) -> SOFTMAX(10)
Test accuracy: 0.9483

Linear decay, relu activations
(CONV3x3x64)x3 -> MAXPOOL -> DROPOUT(0.8) -> (CONV3x3x128)x2 -> CONV1x1x64 -> MAXPOOL -> DROPOUT(0.8) -> FC(2048) -> FC(512) -> FC(256) -> SOFTMAX(10)
Test accuracy: 0.9531

200 iterations, batch_size = 100, Test accuracy: 0.9526001

Linear decay, relu activations, 120 epochs
(CONV3x3x64)x3 -> MAXPOOL -> DROPOUT(0.7) -> (CONV3x3x128)x2 -> CONV1x1x64 -> MAXPOOL -> DROPOUT(0.7) -> FC(2048) -> FC(512) -> FC(256) -> SOFTMAX(10)
Test accuracy: 0.95350003

150 Epochs, batch_size=60 Test accuracy: 0.95369995

Best result with data augmentation: 0.9568
